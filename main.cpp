#include "TextureMapping.h"

int main()
{
	trimesh::TriMesh::set_verbose(0);
	clock_t time = clock();

	//º”‘ÿ ˝æ›
	int frameSize = 11;
	std::vector<trimesh::xform> extrinsics(frameSize);
	std::vector<cv::Mat> imgs(frameSize);
	trimesh::xform leftIntrinsic;
	leftIntrinsic.read("D:\\TextureMapping\\data\\02\\leftIntrinsic.txt");
	std::shared_ptr<trimesh::TriMesh> mesh(trimesh::TriMesh::read("D:\\TextureMapping\\data\\02\\post.ply"));
	trimesh::remove_unused_vertices(mesh.get());
#pragma omp parallel for
	for (int i = 0; i < extrinsics.size(); ++i)
	{
		extrinsics[i].read("D:\\TextureMapping\\data\\02\\" + std::to_string(i) + ".txt");
		trimesh::invert(extrinsics[i]);
		imgs[i] = cv::imread("D:\\TextureMapping\\data\\02\\color_" + std::to_string(i) + ".bmp");
	}
	std::cout << "read time: " << clock() - time << std::endl;
	time = clock();

	cv::Mat atlas;
	std::vector<trimesh::vec2> uvs;
	TextureMapping tm;
	tm.apply(mesh, imgs, extrinsics, leftIntrinsic, atlas, uvs);
	std::cout << "apply time: " << clock() - time << std::endl;
	TextureMapping::writeObj(mesh.get(), atlas, uvs, "D:\\", "output");

	system("pause");
	return 0;
}