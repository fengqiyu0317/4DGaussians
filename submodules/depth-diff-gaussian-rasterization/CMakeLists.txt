#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
	cuda_rasterizer/tacker_forward.cuh
	cuda_rasterizer/tacker_mixed.h
	cuda_rasterizer/tacker_mixed.cu
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
set(TACKER_4DGS_HEAD_INCLUDE
	"${CMAKE_CURRENT_SOURCE_DIR}/../../tacker_ext/include"
	CACHE PATH "Directory containing head_linear_device.cuh")
if(NOT EXISTS "${TACKER_4DGS_HEAD_INCLUDE}/head_linear_device.cuh" OR
	NOT EXISTS "${TACKER_4DGS_HEAD_INCLUDE}/head_linear_v2_device.cuh")
	message(FATAL_ERROR
		"Set TACKER_4DGS_HEAD_INCLUDE to a tacker_ext/include with ABI v1 and v2")
endif()
target_include_directories(CudaRasterizer PRIVATE
	third_party/glm
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${TACKER_4DGS_HEAD_INCLUDE})
