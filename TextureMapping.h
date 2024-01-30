#pragma once
#include <TriMesh.h>
#include <TriMesh_algo.h>

#include <iostream>
#include <opencv2/opencv.hpp>

struct TextureView
{
	int id;
	trimesh::vec3 pos;
	trimesh::vec3 viewdir;
	trimesh::xform projection;
	trimesh::xform world_to_cam;
	cv::Mat image;
	trimesh::vec3 getNormalizedBGR(const trimesh::vec2& coords);
};

struct TexturePatch
{
	int bestView;
	std::vector<int> faces;
	std::vector<trimesh::vec2> texcoords;
	cv::Mat image;
};

struct Tri
{
	trimesh::vec2 v1, v2, v3;
	float min_x, min_y, max_x, max_y, detT;

	Tri(){};

	/** Constructor which calculates the axis aligned bounding box and prepares the calculation of barycentric coordinates. */
	Tri(trimesh::vec2 _v1, trimesh::vec2 _v2, trimesh::vec2 _v3);

	void init(trimesh::vec2 _v1, trimesh::vec2 _v2, trimesh::vec2 _v3);

	/** Determines whether the given point is inside via barycentric coordinates. */
	bool inside(float x, float y);

	/** Returns the barycentric coordinates for the given point. */
	trimesh::vec3 get_barycentric_coords(float x, float y);

	trimesh::vec2 get_pixel_coords(const trimesh::vec3& bcoords);

	/** Returns the area of the triangle. */
	float get_area();
};

struct FaceView
{
	Tri tri;
	int viewId;
	float area = 0.f;
	float cosValue = 0.f;
	std::pair<trimesh::Color, float> color = std::make_pair(trimesh::Color(0.f), 0.f);
	static bool Cmp_Color(const FaceView& a, const FaceView& b)
	{
		return a.color.first.sum() > b.color.first.sum();
	}
	static bool Cmp_Cos(const FaceView& a, const FaceView& b)
	{
		return a.cosValue > b.cosValue;
	}
};

class TextureMapping
{
public:
	TextureMapping(){};

	bool apply(std::shared_ptr<trimesh::TriMesh> mesh,
	           std::vector<cv::Mat> images,
	           std::vector<trimesh::xform> extrinsics,
	           trimesh::xform intrinsic,
	           cv::Mat& atlas,
	           std::vector<trimesh::vec2>& uvs);

	static void writeObj(trimesh::TriMesh* mesh, cv::Mat& atlas, std::vector<trimesh::vec2>& uvs, std::string path, std::string name);

private:
	bool selectBestView(trimesh::TriMesh* mesh, const std::vector<TextureView>& views, std::vector<int>& bestViews);

	bool generatePatch(trimesh::TriMesh* mesh, std::vector<int>& facesViewMap, std::vector<int>& facesPatchMap, std::vector<TexturePatch>& patchs);

	bool globalSeamLeveling(trimesh::TriMesh* mesh, std::vector<TextureView>& views, std::vector<TexturePatch>& patchs, int texture_patch_border);

	bool imageBindinig(trimesh::TriMesh* mesh,
	                   std::vector<TextureView>& views,
	                   std::vector<TexturePatch>& patchs,
	                   int texture_patch_border,
	                   cv::Mat& atlas,
	                   std::vector<trimesh::vec2>& uvs);

	bool packing(trimesh::TriMesh* mesh, std::vector<TexturePatch>& patchs, cv::Mat& atlas, std::vector<trimesh::vec2>& uvs);

private:
	std::shared_ptr<trimesh::TriMesh> mesh_;
	std::vector<TextureView> views_;
	std::vector<int> facesViewMap_;
	std::vector<int> facesPatchMap_;
	std::vector<TexturePatch> patchs_;
	std::vector<std::vector<FaceView>> facesCandidateViews_;
	std::vector<std::vector<float>> vertexWeights_;
};
