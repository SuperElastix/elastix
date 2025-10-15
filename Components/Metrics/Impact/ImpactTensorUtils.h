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
 *
 * \author V. Boussot,  Univ. Rennes, INSERM, LTSI- UMR 1099, F-35000 Rennes, France
 * \note This work was funded by the French National Research Agency as part of the VATSop project (ANR-20-CE19-0015).
 * \note If you use the Impact anywhere we would appreciate if you cite the following article:\n
 * V. Boussot et al., IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration, arXiv preprint
 * arXiv:2503.24121 (2025). https://doi.org/10.48550/arXiv.2503.24121
 *
 */

#ifndef ImpactTensorUtils_h
#define ImpactTensorUtils_h

#include <torch/torch.h>
#include <vector>
#include <functional>
#include <exception>
#include "ImpactLoss.h"
#include "itkImpactModelConfiguration.h"
#include <random>

namespace ImpactTensorUtils
{

/**
 * \brief Converts an ITK image to a Torch tensor using physical spacing.
 *
 * \param voxelSize Target voxel size in mm, used for resampling the image.
 * \param transformPoint Optional point-wise transform to apply to the moving image.
 *
 * \return A Torch tensor representing the resampled ITK image.
 */
template <typename TImage, typename TInterpolator>
torch::Tensor
ImageToTensor(typename TImage::ConstPointer                                                         image,
              typename TInterpolator::Pointer                                                       interpolator,
              const std::vector<float> &                                                            voxelSize,
              const std::function<typename TImage::PointType(const typename TImage::PointType &)> & transformPoint);

/**
 * \brief Converts a tensor (C×D×H×W) to a multi-channel ITK image.
 * Converts the given tensor to an ITK image, preserving the original image metadata (origin, spacing, direction).
 *
 * \param image The original input image used to retrieve metadata.
 * \param layers The tensor representing the feature layers.
 *
 * \return A multi-channel ITK image constructed from the tensor.
 */
template <typename TImage, typename TFeatureImage>
typename TFeatureImage::Pointer
TensorToImage(typename TImage::ConstPointer image, torch::Tensor layers);

/**
 * \brief Applies one or more models to an image to extract feature maps.
 *
 * This function extracts feature maps from an image using one or more models. Optionally, it can
 * export the resampled input image for debugging purposes.
 *
 * \param image The input image from which features are extracted.
 * \param interpolator The interpolator used for resampling.
 * \param modelsConfiguration The configuration of the models used for feature extraction.
 * \param device The device (CPU or GPU) to perform the computation on.
 * \param pca The number of principal components for dimensionality reduction.
 * \param principalComponents A vector to store the computed principal components.
 * \param writeInputImage Optional function to export the resampled input image for debugging.
 * \param transformPoint ptional function to transform a point, used for geometric transformations on the moving image.
 *
 * \return A vector of ITK feature images, one per layer and model.
 */
template <typename TImage, typename FeaturesMaps, typename InterpolatorType, typename FeaturesImageType>
std::vector<FeaturesMaps>
GetFeaturesMaps(
  typename TImage::ConstPointer                                                                    image,
  typename InterpolatorType::Pointer                                                               interpolator,
  const std::vector<itk::ImpactModelConfiguration> &                                               modelsConfiguration,
  torch::Device                                                                                    device,
  std::vector<unsigned int>                                                                        pca,
  std::vector<torch::Tensor> &                                                                     principalComponents,
  const std::function<void(typename TImage::ConstPointer, torch::Tensor &, const std::string &)> & writeInputImage,
  const std::function<typename TImage::PointType(const typename TImage::PointType &)> & transformPoint = nullptr);

/**
 * \brief Tests the configuration of each model by generating outputs from dummy input.
 *
 * This function validates the compatibility of TorchScript models and checks the output structure
 * (e.g., number of layers, spatial shape, channels). It ensures that models are properly loaded
 * and executable during initialization.
 *
 * \param modelsConfig Vector of model configurations.
 * \param modelType The type of the model being tested (fixed, moving) for logging errors.
 * \param device The device (CPU or GPU) to perform the computation on.
 *
 * \return A vector of tensors with the generated outputs from dummy input.
 */
std::vector<torch::Tensor>
GetModelOutputsExample(std::vector<itk::ImpactModelConfiguration> & modelsConfig,
                       const std::string &                          modelType,
                       torch::Device                                device);

/**
 * \brief Computes patch index offsets around a center point based on model configuration.
 *
 * This function generates the offsets for local patches (e.g., 5x5x5) around a center point
 * using the model's configuration. It helps in extracting features from local neighborhoods.
 *
 * \param modelConfiguration Configuration of the model, including patch size and voxel size.
 * \param randomGenerator Random generator used for randomizing 2D patch slices in a 3D volume.
 * \param dimension The image dimension (2D or 3D).
 *
 * \return A vector of offsets for the patch around the center point.
 */
std::vector<std::vector<float>>
GetPatchIndex(const itk::ImpactModelConfiguration & modelConfiguration,
              std::mt19937 &                        randomGenerator,
              unsigned int                          dimension);

template <typename ImagePointType>
using ImagesPatchValuesEvaluator = std::function<
  torch::Tensor(const ImagePointType &, const std::vector<std::vector<float>> &, const std::vector<int64_t> &)>;

/**
 * \brief Computes feature outputs for all patches using each model.
 *
 * This function computes feature outputs for each patch by running a forward pass through the model
 * and applying optional feature sub-selection.
 *
 * \param modelConfig Vector of model configurations (dimensions, patch sizes, voxel sizes, etc.).
 * \param fixedPoints Central points on the fixed image where features are computed.
 * \param patchIndex Indices defining the local patches around each point.
 * \param subsetsOfFeatures Tensors containing indices of feature channels to select.
 * \param device The device (CPU or GPU) to perform the computation on.
 * \param imagesPatchValuesEvaluator  Callable that evaluate image values.
 *
 * \return A vector of tensors containing the computed feature outputs for each patch.
 */
template <class ImagePointType>
std::vector<torch::Tensor>
GenerateOutputs(const std::vector<itk::ImpactModelConfiguration> &                    modelConfig,
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
 * \brief Computes feature outputs and their spatial Jacobians for image registration.
 *
 * This function calculates the feature outputs and their corresponding Jacobians at given points
 * to aid in optimization. The Jacobians are used for gradient computation during image registration.
 *
 * \param modelConfig Vector of model configurations (dimensions, patch sizes, voxel sizes, etc.).
 * \param fixedPoints Central points on the fixed image where features are computed.
 * \param patchIndex  Indices defining the local patches around each point for feature extraction.
 * \param subsetsOfFeatures Tensors containing indices of feature channels to select.
 * \param fixedOutputsTensor Tensor containing feature outputs from the fixed image.
 * \param device The device (CPU or GPU) to perform the computation on.
 * \param losses Vector of loss objects to be updated incrementally during optimization.
 * \param imagesPatchValuesAndJacobiansEvaluator Callable that evaluate image values and Jacobians.
 *
 * \return A vector of tensors with computed feature outputs.
 */
template <typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputsAndJacobian(const std::vector<itk::ImpactModelConfiguration> &                modelConfig,
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
