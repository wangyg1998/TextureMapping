#include "TextureMapping.h"

bool TextureMapping::apply(std::shared_ptr<trimesh::TriMesh> mesh,
                           std::vector<cv::Mat> images,
                           std::vector<trimesh::xform> extrinsics,
                           trimesh::xform intrinsic,
                           cv::Mat& atlas,
                           std::vector<trimesh::vec2>& uvs)
{
	mesh_ = mesh;
	views_.resize(images.size());
	for (int i = 0; i < views_.size(); ++i)
	{
		auto& view = views_[i];
		view.id = i;
		view.world_to_cam = extrinsics[i];
		view.projection = intrinsic * extrinsics[i];
		view.pos = trimesh::inv(view.world_to_cam) * trimesh::point(0.f);
		view.viewdir = trimesh::normalized(view.pos);
		view.image = images[i];
	}
	int texture_patch_border = 3;
	selectBestView(mesh_.get(), views_, facesViewMap_);
	generatePatch(mesh_.get(), facesViewMap_, facesPatchMap_, patchs_);
	imageBindinig(mesh_.get(), views_, patchs_, texture_patch_border, atlas, uvs);
	globalSeamLeveling(mesh_.get(), views_, patchs_, texture_patch_border);
	packing(mesh_.get(), patchs_, atlas, uvs);
	return true;
}

bool TextureMapping::globalSeamLeveling(trimesh::TriMesh* mesh, std::vector<TextureView>& views, std::vector<TexturePatch>& patchs, int texture_patch_border)
{
	clock_t time = clock();

	texture_patch_border = 1;
	const float sqrt_2 = sqrt(2);
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < patchs.size(); ++k)
	{
		TexturePatch& patch = patchs[k];
		cv::Mat& img = patch.image;
		auto& faces = patch.faces;
		int width = patch.image.cols, height = patch.image.rows;
		cv::Mat alreadyUpdated = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
		for (int i = 0; i < faces.size(); ++i)
		{
			int triId = faces[i];
			if (facesCandidateViews_[triId].size() < 2)
			{
				continue;
			}
			const auto& f = mesh->faces[triId];
			const trimesh::vec2& v1 = patch.texcoords[i * 3];
			const trimesh::vec2& v2 = patch.texcoords[i * 3 + 1];
			const trimesh::vec2& v3 = patch.texcoords[i * 3 + 2];
			Tri tri(v1, v2, v3);
			float area = tri.get_area();
			int const min_x = std::max(static_cast<int>(std::floor(tri.min_x)) - texture_patch_border, 0);
			int const min_y = std::max(static_cast<int>(std::floor(tri.min_y)) - texture_patch_border, 0);
			int const max_x = std::min(static_cast<int>(std::ceil(tri.max_x)) + texture_patch_border, width);
			int const max_y = std::min(static_cast<int>(std::ceil(tri.max_y)) + texture_patch_border, height);
			for (int y = min_y; y < max_y; ++y)
			{
				for (int x = min_x; x < max_x; ++x)
				{
					trimesh::vec3 bcoords = tri.get_barycentric_coords(x, y);
					bool inside = bcoords.min() >= 0.0f;
					bool needUpdateColor = false;
					if (inside)
					{
						needUpdateColor = true;
					}
					else
					{
						if (alreadyUpdated.at<uchar>(y, x) == 255)
						{
							continue;
						}

						/* Check whether the pixels distance from the triangle is more than one pixel. */
						float ha = 2.0f * -bcoords[0] * area / trimesh::dist(v2, v3);
						float hb = 2.0f * -bcoords[1] * area / trimesh::dist(v1, v3);
						float hc = 2.0f * -bcoords[2] * area / trimesh::dist(v1, v2);

						if (ha > sqrt_2 || hb > sqrt_2 || hc > sqrt_2)
						{
							continue;
						}
						needUpdateColor = true;
					}
					if (needUpdateColor)
					{
						std::pair<trimesh::vec3, float> weightColor(trimesh::vec3(0.f), 0.f);
						for (auto& candidate : facesCandidateViews_[triId])
						{
							std::vector<float>& vertexWeight = vertexWeights_[candidate.viewId];
							float weight = vertexWeight[f[0]] * bcoords[0] + vertexWeight[f[1]] * bcoords[1] + vertexWeight[f[2]] * bcoords[2];
							weight = std::abs(weight); //规避扩展像素的权重为负的情况
							if (weight < 1e-6f)
							{
								continue;
							}
							trimesh::vec3 color = views[candidate.viewId].getNormalizedBGR(candidate.tri.get_pixel_coords(bcoords));
							weightColor.first += color * weight;
							weightColor.second += weight;
						}
						weightColor.first /= weightColor.second;
						for (int j = 0; j < 3; ++j)
						{
							patch.image.at<cv::Vec3b>(y, x)[j] = weightColor.first[j] * 255.f;
						}
						alreadyUpdated.at<uchar>(y, x) = 255;
					}
				}
			}
		}
	}

	std::cout << "globalSeamLeveling time: " << clock() - time << std::endl;
	return true;
}

bool TextureMapping::packing(trimesh::TriMesh* mesh, std::vector<TexturePatch>& patchs, cv::Mat& atlas, std::vector<trimesh::vec2>& uvs)
{
	int maxHeight = 0, widthSum = 0;
	for (int k = 0; k < patchs.size(); ++k)
	{
		maxHeight = std::max(maxHeight, patchs[k].image.rows);
		widthSum += patchs[k].image.cols;
	}
	atlas = cv::Mat::zeros(maxHeight + 10, widthSum + 10, CV_8UC3);
	int widthAccumulate = 0;
	for (int k = 0; k < patchs.size(); ++k)
	{
		TexturePatch& patch = patchs[k];
		for (auto& uv : patch.texcoords)
		{
			uv.x += widthAccumulate;
			uv.y = atlas.rows - 1 - uv.y;
			uv.x /= static_cast<float>(atlas.cols);
			uv.y /= static_cast<float>(atlas.rows);
		}
		patch.image.copyTo(atlas(cv::Rect(widthAccumulate, 0, patch.image.cols, patch.image.rows)));
		widthAccumulate += patch.image.cols;
	}

	uvs.clear();
	uvs.resize(mesh->faces.size() * 3, trimesh::vec2(0.f));
	for (int k = 0; k < patchs.size(); ++k)
	{
		auto faces = patchs[k].faces;
		for (int i = 0; i < faces.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				uvs[faces[i] * 3 + j] = patchs[k].texcoords[i * 3 + j];
			}
		}
	}
	return true;
}

bool TextureMapping::imageBindinig(trimesh::TriMesh* mesh,
                                   std::vector<TextureView>& views,
                                   std::vector<TexturePatch>& patchs,
                                   int texture_patch_border,
                                   cv::Mat& atlas,
                                   std::vector<trimesh::vec2>& uvs)
{
#pragma omp parallel for
	for (int k = 0; k < patchs.size(); ++k)
	{
		auto& patch = patchs[k];
		//计算纹理坐标
		std::vector<trimesh::vec2>& texcoords = patch.texcoords;
		{
			texcoords.clear();
			texcoords.resize(patch.faces.size() * 3);
			trimesh::xform xf = views[patch.bestView].projection;
			for (int i = 0; i < patch.faces.size(); ++i)
			{
				auto f = mesh->faces[patch.faces[i]];
				for (int j = 0; j < 3; ++j)
				{
					trimesh::point p = xf * mesh->vertices[f[j]];
					texcoords[i * 3 + j] = trimesh::vec2(p.x / p.z, p.y / p.z);
				}
			}
		}

		//图像拷贝
		std::pair<trimesh::ivec2, trimesh::ivec2> imgRange;
		{
			cv::Mat& img = views[patch.bestView].image;
			int width = img.cols, height = img.rows;
			imgRange.first.set(100000);
			imgRange.second.set(-100000);
			for (int i = 0; i < texcoords.size(); ++i)
			{
				for (int j = 0; j < 2; ++j)
				{
					imgRange.first[j] = std::min<int>(imgRange.first[j], texcoords[i][j]);
					imgRange.second[j] = std::max<int>(imgRange.second[j], texcoords[i][j]);
				}
			}
			//边界扩展
			{
				imgRange.first -= texture_patch_border;
				imgRange.second += texture_patch_border;
				for (int j = 0; j < 2; ++j)
				{
					imgRange.first[j] = std::max(imgRange.first[j], 0);
				}
				imgRange.second[0] = std::min(imgRange.second[0], width);
				imgRange.second[1] = std::min(imgRange.second[1], height);
			}
			trimesh::ivec2 imgSize = imgRange.second - imgRange.first;
			patch.image = cv::Mat::zeros(imgSize.y, imgSize.x, CV_8UC3);
			views[patch.bestView].image(cv::Rect(imgRange.first.x, imgRange.first.y, imgSize.x, imgSize.y)).copyTo(patch.image);
		}

		//坐标更新
		{
			for (int i = 0; i < patch.texcoords.size(); ++i)
			{
				patch.texcoords[i].x -= imgRange.first.x;
				patch.texcoords[i].y -= imgRange.first.y;
			}
		}
	}
	return true;
}

bool TextureMapping::generatePatch(trimesh::TriMesh* mesh, std::vector<int>& facesViewMap, std::vector<int>& facesPatchMap, std::vector<TexturePatch>& patchs)
{
	clock_t time = clock();

	patchs.clear();
	//生成patch
	{
		mesh->need_across_edge();
		std::vector<bool> visitedFaces(mesh->faces.size(), false);
		for (int k = 0; k < mesh->faces.size(); ++k)
		{
			if (visitedFaces[k])
			{
				continue;
			}
			TexturePatch patch;
			patch.bestView = facesViewMap[k];
			std::vector<int>& faces = patch.faces;
			faces.push_back(k);
			visitedFaces[k] = true;
			std::pair<int, int> searchRegion(0, faces.size());
			while (searchRegion.first < searchRegion.second)
			{
				for (int i = searchRegion.first; i < searchRegion.second; ++i)
				{
					for (int ring : mesh->across_edge[faces[i]])
					{
						if (ring >= 0 && !visitedFaces[ring] && facesViewMap[ring] == patch.bestView)
						{
							faces.push_back(ring);
							visitedFaces[ring] = true;
						}
					}
				}
				searchRegion.first = searchRegion.second;
				searchRegion.second = faces.size();
			}
			patchs.push_back(patch);
		}
	}
	//合并小patch
	{
		int minPatchSize = 100;
		facesPatchMap.clear();
		facesPatchMap.resize(mesh->faces.size(), -1);
		for (int i = 0; i < patchs.size() - 1; i++)
		{
			for (int j = 0; j < patchs.size() - 1 - i; j++)
			{
				if (patchs[j + 1].faces.size() > patchs[j].faces.size())
				{
					std::swap(patchs[j], patchs[j + 1]);
				}
			}
		}
		std::vector<int> cached;
		while (patchs.back().faces.size() < 100)
		{
			patchs.pop_back();
		}
		for (int k = 0; k < patchs.size(); ++k)
		{
			auto& patch = patchs[k];
			for (int triId : patch.faces)
			{
				facesPatchMap[triId] = k;
			}
			cached.insert(cached.end(), patch.faces.begin(), patch.faces.end());
		}
		std::pair<int, int> searchRegion(0, cached.size());
		while (searchRegion.first < searchRegion.second)
		{
			for (int i = searchRegion.first; i < searchRegion.second; ++i)
			{
				int patchId = facesPatchMap[cached[i]];
				for (int ring : mesh->across_edge[cached[i]])
				{
					if (ring >= 0 && facesPatchMap[ring] < 0)
					{
						facesPatchMap[ring] = patchId;
						patchs[patchId].faces.push_back(ring);
						cached.push_back(ring);
					}
				}
			}
			searchRegion.first = searchRegion.second;
			searchRegion.second = cached.size();
		}
	}
	std::cout << std::endl << "patchs.size(): " << patchs.size() << std::endl;
	//更新映射
	{
		for (int k = 0; k < patchs.size(); ++k)
		{
			int view = patchs[k].bestView;
			for (int triId : patchs[k].faces)
			{
				facesViewMap[triId] = view;
				facesPatchMap[triId] = k;
			}
		}
	}

	std::cout << "generatePatch time: " << clock() - time << std::endl;
	return true;
}

bool TextureMapping::selectBestView(trimesh::TriMesh* mesh, const std::vector<TextureView>& views, std::vector<int>& bestViews)
{
	clock_t time = clock();

	int width = views[0].image.cols, height = views[0].image.rows;
	bestViews.clear();
	bestViews.resize(mesh->faces.size(), -1);

	//按深度进行自遮挡检测<https://www.graphics.pku.edu.cn/docs/20180717164958039986.pdf>
	std::vector<cv::Mat> minDepths(views.size());
#pragma omp parallel for
	for (int k = 0; k < views.size(); ++k)
	{
		cv::Mat& minDepth = minDepths[k];
		minDepth = cv::Mat(width, height, CV_32FC1, 1000000.f);
		trimesh::xform projection = views[k].projection;
		for (int i = 0; i < mesh->faces.size(); ++i)
		{
			auto f = mesh->faces[i];
			trimesh::vec2 v[3];
			trimesh::vec3 depth;
			for (int j = 0; j < 3; ++j)
			{
				trimesh::point p = projection * mesh->vertices[f[j]];
				depth[j] = p.z;
				v[j].x = p.x / p.z; //[https://ww2.mathworks.cn/help/images/image-coordinate-systems.html]
				v[j].y = p.y / p.z;
			}
			Tri tri(v[0], v[1], v[2]);
			if (tri.min_x > 1e-6f && tri.min_y > 1e-6f && tri.max_x < width && tri.max_y < height)
			{
				int const min_x = static_cast<int>(std::floor(tri.min_x));
				int const min_y = static_cast<int>(std::floor(tri.min_y));
				int const max_x = static_cast<int>(std::ceil(tri.max_x));
				int const max_y = static_cast<int>(std::ceil(tri.max_y));
				for (int y = min_y; y < max_y; ++y)
				{
					for (int x = min_x; x < max_x; ++x)
					{
						trimesh::vec3 bcoords = tri.get_barycentric_coords(x, y);
						bool inside = bcoords.min() >= 0.0f;
						if (inside)
						{
							float nowDepth = depth.dot(bcoords);
							if (minDepth.at<float>(y, x) > nowDepth)
							{
								minDepth.at<float>(y, x) = nowDepth;
							}
						}
					}
				}
			}
		}
	}

	//筛选面片最佳视角--剔除阴影和高光
	{
		facesCandidateViews_.resize(mesh->faces.size());
		std::atomic<int> errorViewCount = 0;
#pragma omp parallel for
		for (int i = 0; i < mesh->faces.size(); ++i)
		{
			auto f = mesh->faces[i];
			trimesh::point normal = mesh->trinorm(i);
			trimesh::normalize(normal);
			std::vector<FaceView>& candidates = facesCandidateViews_[i];
			for (int j = 0; j < views.size(); ++j)
			{
				if (normal.dot(views[j].viewdir) < 0.f)
				{
					continue;
				}
				int valid = 0;
				trimesh::ivec2 min(INT_MAX), max(-INT_MAX);
				trimesh::vec2 v[3];
				for (int k = 0; k < 3; ++k)
				{
					trimesh::point p = views[j].projection * mesh->vertices[f[k]];
					float t_depth = p.z;
					v[k].x = p.x / p.z;
					v[k].y = p.y / p.z;
					int x = v[k].x, y = v[k].y;
					if (x >= 0 && x < width && y >= 0 && y < height && std::abs(minDepths[j].at<float>(y, x) - t_depth) < 1.f)
					{
						min.x = std::min(min.x, x);
						max.x = std::max(max.x, x);
						min.y = std::min(min.y, y);
						max.y = std::max(max.y, y);
						valid += 1;
					}
				}
				if (valid == 3)
				{
					FaceView candidate;
					candidate.viewId = j;
					candidate.tri.init(v[0], v[1], v[2]);
					candidate.area = candidate.tri.get_area();
					candidate.cosValue = normal.dot(views[j].viewdir);
					for (int x = min.x; x <= max.x; ++x)
					{
						for (int y = min.y; y <= max.y; ++y)
						{
							auto bgr = views[j].image.at<cv::Vec3b>(y, x);
							candidate.color.first += trimesh::Color(bgr[2], bgr[1], bgr[0]);
							candidate.color.second += 1;
						}
					}
					candidate.color.first /= candidate.color.second;
					candidates.push_back(candidate);
				}
			}
			if (candidates.empty())
			{
				errorViewCount++;
				bestViews[i] = 0;
				for (int j = 0; j < views.size(); ++j)
				{
					if (normal.dot(views[j].viewdir) > normal.dot(views[bestViews[i]].viewdir))
					{
						bestViews[i] = j;
					}
				}
			}
			else if (candidates.size() == 1)
			{
				bestViews[i] = candidates[0].viewId;
			}
			else
			{
				std::sort(candidates.begin(), candidates.end(), FaceView::Cmp_Cos);
				bestViews[i] = candidates[0].viewId;
			}
		}
		if (errorViewCount > 0)
		{
			std::cout << "errorViewCount: " << errorViewCount << std::endl;
		}
	}

	mesh->need_neighbors();
	mesh->need_adjacentfaces();
	vertexWeights_.clear();
	vertexWeights_.resize(views.size(), std::vector<float>(mesh->vertices.size(), 0.f));

	//面积权重
	std::vector<std::vector<float>> facesAreaWeights(views.size(), std::vector<float>(mesh->faces.size(), 0.f));
	for (int i = 0; i < facesCandidateViews_.size(); ++i)
	{
		for (const auto& faceView : facesCandidateViews_[i])
		{
			facesAreaWeights[faceView.viewId][i] = faceView.area;
		}
	}

	//边界权重
	std::vector<std::vector<float>> borderWeights(views.size(), std::vector<float>(mesh->vertices.size(), 10000.f));
	{
		for (int i = 0; i < facesCandidateViews_.size(); ++i)
		{
			auto f = mesh->faces[i];
			for (const auto& faceView : facesCandidateViews_[i])
			{
				for (int j = 0; j < 3; ++j)
				{
					borderWeights[faceView.viewId][f[j]] = -10000.f;
				}
			}
		}
		for (int k = 0; k < borderWeights.size(); ++k)
		{
			std::vector<float>& weight = borderWeights[k];
			std::vector<int> cached;
			for (int i = 0; i < mesh->vertices.size(); ++i)
			{
				if (weight[i] < 0.f)
				{
					for (int ring : mesh->neighbors[i])
					{
						if (weight[ring] > 0.f)
						{
							cached.push_back(i);
							weight[i] = 1.f;
							break;
						}
					}
				}
			}
			std::pair<int, int> searchRegion(0, cached.size());
			while (searchRegion.first < searchRegion.second)
			{
				for (int i = searchRegion.first; i < searchRegion.second; ++i)
				{
					int vid = cached[i];
					for (int ring : mesh->neighbors[vid])
					{
						if (weight[ring] < 0.f)
						{
							weight[ring] = weight[vid] + trimesh::dist(mesh->vertices[vid], mesh->vertices[ring]);
							cached.push_back(ring);
						}
					}
				}
				searchRegion.first = searchRegion.second;
				searchRegion.second = cached.size();
			}

			float maxWeight = 0.f;
			float threshold = 1000.f;
			for (int i = 0; i < weight.size(); ++i)
			{
				if (weight[i] > threshold)
				{
					weight[i] = 0.f;
				}
				else
				{
					maxWeight = std::max(maxWeight, weight[i]);
				}
			}
		}
	}

	//最终权重
	for (int k = 0; k < views.size(); ++k)
	{
		std::vector<float>& areaWeight = facesAreaWeights[k];
		std::vector<float>& borderWeight = borderWeights[k];
		for (int i = 0; i < mesh->vertices.size(); ++i)
		{
			if (mesh->adjacentfaces[i].empty())
			{
				continue;
			}
			float vertexAreaWeight = 0.f;
			for (int ring : mesh->adjacentfaces[i])
			{
				vertexAreaWeight += areaWeight[ring];
			}
			vertexAreaWeight /= static_cast<float>(mesh->adjacentfaces[i].size());
			vertexWeights_[k][i] = vertexAreaWeight * borderWeight[i];
		}
	}

	std::cout << "selectBestView time: " << clock() - time << std::endl;
	return true;
}

void TextureMapping::writeObj(trimesh::TriMesh* mesh, cv::Mat& atlas, std::vector<trimesh::vec2>& uvs, std::string path, std::string name)
{
	cv::imwrite(path + name + ".png", atlas);
	std::ofstream out(path + name + ".obj", std::ios::binary);
	out << "mtllib " + name + ".mtl\n";

	for (int i = 0; i < mesh->vertices.size(); i++)
	{
		out << "v " + std::to_string(mesh->vertices[i].x) + " " + std::to_string(mesh->vertices[i].y) + " " + std::to_string(mesh->vertices[i].z) + "\n";
	}
	for (auto uv : uvs)
	{
		out << "vt " + std::to_string(uv.x) + " " + std::to_string(uv.y) + "\n";
	}
	out << "g Group_Global\ns off\nusemtl " + name + "\n";
	for (int i = 0; i < mesh->faces.size(); ++i)
	{
		auto f = mesh->faces[i];
		out << "f";
		for (int j = 0; j < 3; ++j)
		{
			out << " " + std::to_string(f[j] + 1) + "/" + std::to_string(i * 3 + j + 1);
		}
		out << "\n";
	}
	out.close();
	out.open(path + name + ".mtl");
	out << "newmtl " + name + "\n";
	out << "Kd 1.0 1.0 1.0\nKa 0.0 0.0 0.0\nKs 0.0 0.0 0.0\nd 1.0\nNs 0.0\nillum 0\nmap_Kd " + name + ".png\n";
	out << "# end of file\n";
	out.close();
	return;
}

trimesh::vec3 TextureView::getNormalizedBGR(const trimesh::vec2& coords)
{
	float x = std::max(0.0f, std::min(static_cast<float>(image.cols - 1), coords.x));
	float y = std::max(0.0f, std::min(static_cast<float>(image.rows - 1), coords.y));
	cv::Mat patch;
	cv::getRectSubPix(image, cv::Size(1, 1), cv::Point2f(x, y), patch);
	auto bgr = patch.at<cv::Vec3b>(0, 0);
	return trimesh::vec3(bgr[0] / 255.f, bgr[1] / 255.f, bgr[2] / 255.f);
}

void Tri::init(trimesh::vec2 _v1, trimesh::vec2 _v2, trimesh::vec2 _v3)
{
	v1 = _v1;
	v2 = _v2;
	v3 = _v3;
	trimesh::vec4 T;
	T[0] = v1[0] - v3[0];
	T[1] = v2[0] - v3[0];
	T[2] = v1[1] - v3[1];
	T[3] = v2[1] - v3[1];

	detT = T[0] * T[3] - T[2] * T[1];

	//axis aligned bounding box
	min_x = std::min(v1[0], std::min(v2[0], v3[0]));
	min_y = std::min(v1[1], std::min(v2[1], v3[1]));
	max_x = std::max(v1[0], std::max(v2[0], v3[0]));
	max_y = std::max(v1[1], std::max(v2[1], v3[1]));

	return;
}

Tri::Tri(trimesh::vec2 _v1, trimesh::vec2 _v2, trimesh::vec2 _v3)
{
	init(_v1, _v2, _v3);
};

bool Tri::inside(float x, float y)
{
	float const dx = (x - v3[0]);
	float const dy = (y - v3[1]);

	float const alpha = ((v2[1] - v3[1]) * dx + (v3[0] - v2[0]) * dy) / detT;
	if (alpha < 0.0f || alpha > 1.0f)
		return false;

	float const beta = ((v3[1] - v1[1]) * dx + (v1[0] - v3[0]) * dy) / detT;
	if (beta < 0.0f || beta > 1.0f)
		return false;

	if (alpha + beta > 1.0f)
		return false;

	/* else */
	return true;
};

trimesh::vec2 Tri::get_pixel_coords(const trimesh::vec3& bcoords)
{
	return v1 * bcoords.x + v2 * bcoords.y + v3 * bcoords.z;
}

trimesh::vec3 Tri::get_barycentric_coords(float x, float y)
{
	float const alpha = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / detT;
	float const beta = ((v3[1] - v1[1]) * (x - v3[0]) + (v1[0] - v3[0]) * (y - v3[1])) / detT;
	float const gamma = 1.0f - alpha - beta;
	return trimesh::vec3(alpha, beta, gamma);
};

float Tri::get_area()
{
	trimesh::vec2 u = v2 - v1;
	trimesh::vec2 v = v3 - v1;
	return 0.5f * std::abs(u[0] * v[1] - u[1] * v[0]);
};