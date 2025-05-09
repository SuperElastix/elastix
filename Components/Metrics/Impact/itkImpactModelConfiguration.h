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
                           bool                      is_static,
                           bool                      useMixedPrecision)
    : m_modelPath(modelPath)
    , m_dimension(dimension)
    , m_numberOfChannels(numberOfChannels)
    , m_patchSize(patchSize.begin(), patchSize.end())
    , m_voxelSize(voxelSize)
    , m_layersMask(layersMask)
    , m_dtype(useMixedPrecision ? torch::kFloat16 : torch::kFloat32)
  {
    this->m_model =
      std::make_shared<torch::jit::script::Module>(torch::jit::load(this->m_modelPath, torch::Device(torch::kCPU)));
    this->m_model->eval();
    this->m_model->to(torch::kFloat);
    if (!is_static)
    {
      /** Initialize some variables precalculation for loop performance */
      this->m_patchIndex.clear();
      if (this->m_patchSize.size() == 2)
      {
        for (int y = 0; y < this->m_patchSize[1]; ++y)
        {
          for (int x = 0; x < this->m_patchSize[0]; ++x)
          {
            this->m_patchIndex.push_back({ (x - this->m_patchSize[0] / 2) * this->m_voxelSize[0],
                                           (y - this->m_patchSize[1] / 2) * this->m_voxelSize[1] });
          }
        }
      }
      else
      {
        for (int z = 0; z < this->m_patchSize[2]; ++z)
        {
          for (int y = 0; y < this->m_patchSize[1]; ++y)
          {
            for (int x = 0; x < this->m_patchSize[0]; ++x)
            {
              this->m_patchIndex.push_back({ (x - this->m_patchSize[0] / 2) * this->m_voxelSize[0],
                                             (y - this->m_patchSize[1] / 2) * this->m_voxelSize[1],
                                             (z - this->m_patchSize[2] / 2) * this->m_voxelSize[2] });
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
    return m_modelPath == rhs.m_modelPath && m_dimension == rhs.m_dimension &&
           m_numberOfChannels == rhs.m_numberOfChannels && m_patchSize == rhs.m_patchSize &&
           m_voxelSize == rhs.m_voxelSize && m_layersMask == rhs.m_layersMask;
  }

  friend std::ostream &
  operator<<(std::ostream & os, const ImpactModelConfiguration & config)
  {
    os << "\t\tPath : " << config.m_modelPath << "\n\t\tDimension : " << config.m_dimension
       << "\n\t\tNumberOfChannels : " << config.m_numberOfChannels
       << "\n\t\tPatchSize : " << GetStringFromVector<int64_t>(config.m_patchSize)
       << "\n\t\tVoxelSize : " << GetStringFromVector<float>(config.m_voxelSize)
       << "\n\t\tLayersMask : " << GetStringFromVector<bool>(config.m_layersMask);
    return os;
  }

  const std::string &
  GetModelPath() const
  {
    return m_modelPath;
  }

  const torch::ScalarType &
  Getdtype() const
  {
    return m_dtype;
  }

  unsigned int
  GetDimension() const
  {
    return m_dimension;
  }
  unsigned int
  GetNumberOfChannels() const
  {
    return m_numberOfChannels;
  }
  const std::vector<int64_t> &
  GetPatchSize() const
  {
    return m_patchSize;
  }
  const std::vector<float> &
  GetVoxelSize() const
  {
    return m_voxelSize;
  }
  const std::vector<bool> &
  GetLayersMask() const
  {
    return m_layersMask;
  }
  torch::jit::script::Module &
  GetModel() const
  {
    return *m_model;
  }
  const std::vector<std::vector<float>> &
  GetPatchIndex() const
  {
    return m_patchIndex;
  }
  const std::vector<std::vector<torch::indexing::TensorIndex>> &
  GetCentersIndexLayers() const
  {
    return m_centersIndexLayers;
  }
  void
  SetCentersIndexLayers(std::vector<std::vector<torch::indexing::TensorIndex>> & centersIndexLayers)
  {
    this->m_centersIndexLayers = centersIndexLayers;
  }


private:
  std::string                                            m_modelPath;
  unsigned int                                           m_dimension;
  unsigned int                                           m_numberOfChannels;
  std::vector<int64_t>                                   m_patchSize;
  std::vector<float>                                     m_voxelSize;
  std::vector<bool>                                      m_layersMask;
  std::shared_ptr<torch::jit::script::Module>            m_model;
  std::vector<std::vector<float>>                        m_patchIndex;
  std::vector<std::vector<torch::indexing::TensorIndex>> m_centersIndexLayers;
  torch::ScalarType                                      m_dtype;
};


} // end namespace itk

#endif // end #ifndef itkImpactModelConfiguration_h
