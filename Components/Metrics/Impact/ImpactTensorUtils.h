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

/**
 * \file ImpactTensorUtils.h
 * \brief Utilities for converting ITK images to Torch tensors and extracting features using TorchScript models.
 *
 * This module supports:
 *  - Conversion between ITK image data and Torch tensors
 *  - Patch-based evaluation of feature models (optionally with Jacobians)
 *  - Optional PCA projection and feature selection
 *  - Export of features and inputs for inspection
 *
 * These tools are used internally by the ImpactImageToImageMetric for deep learning-based image registration.
 */

#ifndef ImpactTensorUtils_h
#define ImpactTensorUtils_h

#include <torch/torch.h>
#include <vector>
#include <functional>
#include <exception>
#include "ImpactLoss.h"
#include <random>

namespace ImpactTensorUtils
{

/**
 * \brief Converts an ITK image to a Torch tensor using physical spacing.
 * \param voxelSize Target voxel size in mm (used to resample).
 * \param transformPoint Optional point-wise transform (e.g., deformation field).
 */
template <typename TImage, typename TInterpolator>
torch::Tensor
ImageToTensor(typename TImage::ConstPointer                                                         image,
              typename TInterpolator::Pointer                                                       interpolator,
              const std::vector<float> &                                                            voxelSize,
              const std::function<typename TImage::PointType(const typename TImage::PointType &)> & transformPoint);

/**
 * \brief Converts a tensor (C×D×H×W) to a multi-channel ITK image.
 * \details Uses the original image metadata (origin, spacing, direction).
 */
template <typename TImage, typename TFeatureImage>
typename TFeatureImage::Pointer
TensorToImage(typename TImage::ConstPointer image, torch::Tensor layers);

/**
 * \brief Applies one or more models to an image to extract feature maps.
 * \param writeInputImage Optional function to export resampled input for debugging.
 * \return Vector of ITK feature images, one per layer and model.
 */
template <typename TImage,
          typename FeaturesMaps,
          typename InterpolatorType,
          typename ModelConfiguration,
          typename FeaturesImageType>
std::vector<FeaturesMaps>
GetFeaturesMaps(
  typename TImage::ConstPointer                                                                    image,
  typename InterpolatorType::Pointer                                                               interpolator,
  const std::vector<ModelConfiguration> &                                                          modelsConfiguration,
  torch::Device                                                                                    device,
  std::vector<unsigned int>                                                                        pca,
  std::vector<torch::Tensor> &                                                                     principal_components,
  const std::function<void(typename TImage::ConstPointer, torch::Tensor &, const std::string &)> & writeInputImage,
  const std::function<typename TImage::PointType(const typename TImage::PointType &)> & transformPoint = nullptr);

/**
 * \brief Tests the configuration of each model by generating outputs from dummy input.
 *
 * This is useful for validating TorchScript model compatibility and inferring output structure
 * (e.g., number of layers, spatial shape, channels).
 *
 * Called during initialization to ensure models are properly loaded and executable.
 */
template <typename ModelConfiguration>
std::vector<torch::Tensor>
GetModelOutputsExample(std::vector<ModelConfiguration> & modelsConfig,
                       const std::string &               modelType,
                       torch::Device                     device);

/**
 * \brief Computes patch index offsets around a center point based on model config.
 *
 * This is used to extract local neighborhoods for each model (e.g., 5x5x5 patch).
 */
template <typename ModelConfiguration>
std::vector<std::vector<float>>
GetPatchIndex(ModelConfiguration modelConfiguration, std::mt19937 & randomGenerator, unsigned int dimension);

template <typename ImagePointType>
using ImagesPatchValuesEvaluator = std::function<
  torch::Tensor(const ImagePointType &, const std::vector<std::vector<float>> &, const std::vector<int64_t> &)>;

/**
 * \brief Computes feature outputs for all patches using each model.
 *
 * For each patch:
 *   - runs model forward pass
 *   - applies optional feature sub-selection
 *
 * \param evaluator  Callable to produce a tensor from a point + patch + subset
 */
template <class ModelConfiguration, class ImagePointType>
std::vector<torch::Tensor>
GenerateOutputs(const std::vector<ModelConfiguration> &                               modelConfig,
                const std::vector<ImagePointType> &                                   fixedPoints,
                const std::vector<std::vector<std::vector<std::vector<float>>>> &     patchIndex,
                const std::vector<torch::Tensor>                                      subsetsOfFeatures,
                torch::Device                                                         device,
                const ImpactTensorUtils::ImagesPatchValuesEvaluator<ImagePointType> & imagesPatchValuesEvaluator);

template <typename ImagePointType>
using ImagesPatchValuesAndJacobiansEvaluator = std::function<torch::Tensor(const ImagePointType &,
                                                                           torch::Tensor &,
                                                                           const std::vector<std::vector<float>> &,
                                                                           const std::vector<int64_t> &,
                                                                           int)>;

/**
 * \brief Computes both feature outputs and their spatial Jacobians.
 *
 * For use in Jacobian-based optimization mode. Requires backpropagation through TorchScript.
 *
 * \param evaluator Callable that returns a pair (features, jacobian) from a point.
 * \param losses    Mutable references to per-layer loss objects (updated incrementally).
 */
template <typename ModelConfiguration, typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputsAndJacobian(const std::vector<ModelConfiguration> &                           modelConfig,
                           const std::vector<ImagePointType> &                               fixedPoints,
                           const std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex,
                           std::vector<torch::Tensor>                                        subsetsOfFeatures,
                           std::vector<torch::Tensor>                                        fixedOutputsTensor,
                           torch::Device                                                     device,
                           std::vector<std::unique_ptr<ImpactLoss::Loss>> &                  losses,
                           const ImpactTensorUtils::ImagesPatchValuesAndJacobiansEvaluator<ImagePointType> &
                             imagesPatchValuesAndJacobiansEvaluator);

} // namespace ImpactTensorUtils


#ifndef ITK_MANUAL_INSTANTIATION
#  include "ImpactTensorUtils.hxx"
#endif

#endif // end #ifndef ImpactTensorUtils_h
