#pragma once

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 

#include "baked_model.hpp"


struct IndexedMesh
{
	std::uint32_t materialId;
	std::uint32_t indexSize;
	bool isAlphaMask;
	bool isNormalMap;

	labutils::Buffer pos;
	labutils::Buffer texcoords;
	labutils::Buffer normals;
	labutils::Buffer indices;
	labutils::Buffer tangent;
	labutils::Buffer packedTBN;

	//Default constructor
	IndexedMesh(labutils::Buffer pPos, labutils::Buffer pTexCoord, labutils::Buffer pNormal,
		labutils::Buffer pIndices, std::uint32_t pMaterialId, std::uint32_t pIndexSize,bool isAlphaMask, bool isNormalMap
	, labutils::Buffer pTangent, labutils::Buffer ppackedTBN)
		:pos(std::move(pPos)),texcoords(std::move(pTexCoord)),normals(std::move(pNormal)),
		indices(std::move(pIndices)),materialId(pMaterialId),indexSize(pIndexSize), isAlphaMask(isAlphaMask),isNormalMap(isNormalMap),
		tangent(std::move(pTangent)),packedTBN(std::move(ppackedTBN))
	{}


	IndexedMesh(IndexedMesh&& other)noexcept :
		pos(std::move(other.pos)), texcoords(std::move(other.texcoords)), normals(std::move(other.normals)),
		indices(std::move(other.indices)), materialId(other.materialId), indexSize(other.indexSize), isAlphaMask(other.isAlphaMask), isNormalMap(other.isNormalMap),
		tangent(std::move(other.tangent)),packedTBN(std::move(other.packedTBN))
	{}
};

IndexedMesh create_indexed_mesh(labutils::VulkanContext const&, labutils::Allocator const&, BakedModel const&,std::uint32_t meshIndex);