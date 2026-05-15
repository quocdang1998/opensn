// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/inline_macro.h"
#include <array>
#include <cstdint>

namespace opensn
{

enum FaceViewGetter : unsigned int
{
  None = 0,
  WithMSurf = 1,
  WithSurfIntegrals = 2
};

struct FaceViewStaticData
{
  __inline_host_dev__ FaceViewStaticData() {}

  std::uint32_t num_face_nodes;
  std::uint32_t face_node_offset;
  double* outflow;
  std::array<double, 3> normal;
};

/// Face view from contiguous block of memory
struct FaceView : public FaceViewStaticData
{
  __inline_host_dev__ FaceView() {}

  template <FaceViewGetter g = FaceViewGetter::None>
  __inline_host_dev__ const double* Update(const char* face_data)
  {
    const FaceViewStaticData* face_static_data =
      reinterpret_cast<const FaceViewStaticData*>(face_data);
    static_cast<FaceViewStaticData&>(*this) = *(face_static_data++);
    face_data = reinterpret_cast<const char*>(face_static_data);

    const double* result = nullptr;
    if constexpr (g == FaceViewGetter::WithMSurf)
      result = reinterpret_cast<const double*>(face_data);
    else if constexpr (g == FaceViewGetter::WithSurfIntegrals)
      result = reinterpret_cast<const double*>(face_data) + num_face_nodes * num_face_nodes;

    face_data += (num_face_nodes * num_face_nodes + num_face_nodes) * sizeof(double);
    cell_mapping_data = reinterpret_cast<const std::uint32_t*>(face_data);
    return result;
  }

  const std::uint32_t* cell_mapping_data;
};

/// Cell view from contiguous block of memory
struct CellView
{
  __inline_host_dev__ CellView() {}
  __inline_host_dev__ void Update(const char* cell_data)
  {
    // number of faces and nodes (num_node = num_dof!)
    const std::uint32_t* num_node_and_face_data = reinterpret_cast<const std::uint32_t*>(cell_data);
    num_nodes = *(num_node_and_face_data++);
    num_faces = *(num_node_and_face_data++);
    cell_data = reinterpret_cast<const char*>(num_node_and_face_data);
    // total cross section pointer
    const double* const* total_xs_data = reinterpret_cast<const double* const*>(cell_data);
    total_xs = *(total_xs_data++);
    cell_data = reinterpret_cast<const char*>(total_xs_data);
    // phi address
    const std::uint64_t* phi_address_data = reinterpret_cast<const std::uint64_t*>(cell_data);
    phi_address = *(phi_address_data++);
    cell_data = reinterpret_cast<const char*>(phi_address_data);
    // save psi index
    const std::uint64_t* save_psi_index_data = reinterpret_cast<const std::uint64_t*>(cell_data);
    save_psi_index = *(save_psi_index_data++);
    cell_data = reinterpret_cast<const char*>(save_psi_index_data);
    // GM matrix
    GM_data = reinterpret_cast<const double*>(cell_data);
    cell_data = reinterpret_cast<const char*>(GM_data + num_nodes * num_nodes * 4);
    // face data
    offset_face_data = reinterpret_cast<const std::uint64_t*>(cell_data);
    face_data = reinterpret_cast<const char*>(offset_face_data + num_faces);
  }

  template <FaceViewGetter g = FaceViewGetter::None>
  __inline_host_dev__ const double* GetFaceView(FaceView& face, const std::uint32_t& face_index)
  {
    return face.Update<g>(face_data + offset_face_data[face_index]);
  }

  std::uint32_t num_nodes;
  std::uint32_t num_faces;
  const double* total_xs;
  std::uint64_t phi_address;
  std::uint64_t save_psi_index;
  const double* GM_data;
  const std::uint64_t* offset_face_data;
  const char* face_data;
};

/// Mesh view from contiguous block of memory
struct MeshView
{
  __inline_host_dev__ MeshView(const char* mesh_data)
  {
    const std::uint64_t* num_cells_data = reinterpret_cast<const std::uint64_t*>(mesh_data);
    num_cells = *(num_cells_data++);
    offset_cell_data = num_cells_data;
    cell_data = reinterpret_cast<const char*>(offset_cell_data + num_cells);
  }

  __inline_host_dev__ void GetCellView(CellView& cell, const std::uint32_t& cell_index)
  {
    cell.Update(cell_data + offset_cell_data[cell_index]);
  }

  const char* cell_data;
  std::uint64_t num_cells;
  const std::uint64_t* offset_cell_data;
};

} // namespace opensn
