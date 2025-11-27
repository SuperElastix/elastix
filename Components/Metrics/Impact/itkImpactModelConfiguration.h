/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkImpactModelConfiguration_h
#define itkImpactModelConfiguration_h

#include <torch/script.h>
#include <torch/torch.h>

// Standard C++ header files:
#include <algorithm> // For transform.
#include <memory>    // For shared_ptr.
#include <sstream>
#include <string>
#include <vector>

/**
 * ******************* GetStringFromVector ***********************
 */
template <typename T>
std::string
GetStringFromVector(const std::vector<T> & vec)
{
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < vec.size(); ++i)
  {
    ss << vec[i];
    if (i != vec.size() - 1)
    {
      ss << " ";
    }
  }
  ss << ")";
  return ss.str();
} // end GetStringFromVector


namespace itk
{
/**
 * Configuration structure for a TorchScript model used to extract semantic features.
 *
 * Contains path to the model, number of input channels, patch size and voxel size,
 * along with internal buffers (e.g., precomputed patch index and center extraction index).
 * If the mode is not static, the patchIndex is generated here to optimize runtime computation.
 */
struct ImpactModelConfiguration
{
public:
  ImpactModelConfiguration(std::string               modelPath,
                           unsigned int              dimension,
                           unsigned int              numberOfChannels,
                           std::vector<unsigned int> patchSize,
                           std::vector<float>        voxelSize,
                           std::vector<bool>         layersMask,
                           bool                      isStatic,
                           bool                      useMixedPrecision)
    : m_ModelPath(modelPath)
    , m_Dimension(dimension)
    , m_NumberOfChannels(numberOfChannels)
    , m_PatchSize(patchSize.begin(), patchSize.end())
    , m_VoxelSize(voxelSize)
    , m_LayersMask(layersMask)
    , m_DataType(useMixedPrecision ? torch::kFloat16 : torch::kFloat32)
  {
    m_Model = std::make_shared<torch::jit::script::Module>(torch::jit::load(m_ModelPath, torch::Device(torch::kCPU)));
    m_Model->eval();
    m_Model->to(m_DataType);
    if (!isStatic)
    {
      /** Initialize some variables precalculation for loop performance */
      m_PatchIndex.clear();
      if (m_PatchSize.size() == 2)
      {
        for (int y = 0; y < m_PatchSize[1]; ++y)
        {
          for (int x = 0; x < m_PatchSize[0]; ++x)
          {
            m_PatchIndex.push_back(
              { (x - m_PatchSize[0] / 2) * m_VoxelSize[0], (y - m_PatchSize[1] / 2) * m_VoxelSize[1] });
          }
        }
      }
      else
      {
        for (int z = 0; z < m_PatchSize[2]; ++z)
        {
          for (int y = 0; y < m_PatchSize[1]; ++y)
          {
            for (int x = 0; x < m_PatchSize[0]; ++x)
            {
              m_PatchIndex.push_back({ (x - m_PatchSize[0] / 2) * m_VoxelSize[0],
                                       (y - m_PatchSize[1] / 2) * m_VoxelSize[1],
                                       (z - m_PatchSize[2] / 2) * m_VoxelSize[2] });
            }
          }
        }
      }
    }
  }

  // Disable (delete) copying, to avoid having multiple copies of the same model:
  ImpactModelConfiguration(const ImpactModelConfiguration &) = delete;
  ImpactModelConfiguration &
  operator=(const ImpactModelConfiguration &) = delete;

  // Enable (default) move semantics:
  ImpactModelConfiguration(ImpactModelConfiguration &&) = default;
  ImpactModelConfiguration &
  operator=(ImpactModelConfiguration &&) = default;

  // Destructor.
  ~ImpactModelConfiguration() = default;

  bool
  operator==(const ImpactModelConfiguration & rhs) const
  {
    return m_ModelPath == rhs.m_ModelPath && m_Dimension == rhs.m_Dimension &&
           m_NumberOfChannels == rhs.m_NumberOfChannels && m_PatchSize == rhs.m_PatchSize &&
           m_VoxelSize == rhs.m_VoxelSize && m_LayersMask == rhs.m_LayersMask;
  }

  friend std::ostream &
  operator<<(std::ostream & os, const ImpactModelConfiguration & config)
  {
    os << "\t\tPath : " << config.m_ModelPath << "\n\t\tDimension : " << config.m_Dimension
       << "\n\t\tNumberOfChannels : " << config.m_NumberOfChannels
       << "\n\t\tPatchSize : " << GetStringFromVector<int64_t>(config.m_PatchSize)
       << "\n\t\tVoxelSize : " << GetStringFromVector<float>(config.m_VoxelSize)
       << "\n\t\tLayersMask : " << GetStringFromVector<bool>(config.m_LayersMask);
    return os;
  }

  const std::string &
  GetModelPath() const
  {
    return m_ModelPath;
  }

  const torch::ScalarType &
  GetDataType() const
  {
    return m_DataType;
  }

  unsigned int
  GetDimension() const
  {
    return m_Dimension;
  }
  unsigned int
  GetNumberOfChannels() const
  {
    return m_NumberOfChannels;
  }
  const std::vector<int64_t> &
  GetPatchSize() const
  {
    return m_PatchSize;
  }
  const std::vector<float> &
  GetVoxelSize() const
  {
    return m_VoxelSize;
  }
  const std::vector<bool> &
  GetLayersMask() const
  {
    return m_LayersMask;
  }
  torch::jit::script::Module &
  GetModel() const
  {
    return *m_Model;
  }
  const std::vector<std::vector<float>> &
  GetPatchIndex() const
  {
    return m_PatchIndex;
  }
  const std::vector<std::vector<torch::indexing::TensorIndex>> &
  GetCentersIndexLayers() const
  {
    return m_CentersIndexLayers;
  }
  void
  SetCentersIndexLayers(std::vector<std::vector<torch::indexing::TensorIndex>> & centersIndexLayers)
  {
    m_CentersIndexLayers = centersIndexLayers;
  }


private:
  std::string                                            m_ModelPath;
  unsigned int                                           m_Dimension;
  unsigned int                                           m_NumberOfChannels;
  std::vector<int64_t>                                   m_PatchSize;
  std::vector<float>                                     m_VoxelSize;
  std::vector<bool>                                      m_LayersMask;
  std::shared_ptr<torch::jit::script::Module>            m_Model;
  std::vector<std::vector<float>>                        m_PatchIndex;
  std::vector<std::vector<torch::indexing::TensorIndex>> m_CentersIndexLayers;
  torch::ScalarType                                      m_DataType;
};


} // end namespace itk

#endif // end #ifndef itkImpactModelConfiguration_h
