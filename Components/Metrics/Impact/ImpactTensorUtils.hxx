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

#ifndef _ImpactTensorUtils_hxx
#define _ImpactTensorUtils_hxx

#include "ImpactTensorUtils.h"
#include "elxlog.h"
#include <ATen/autocast_mode.h>

/**
 * ImageToTensor: Converts ITK image to torch tensor with spatial resampling
 *
 * @tparam TImage ITK image type with GetLargestPossibleRegion(), GetSpacing(), GetOrigin()
 * @tparam TInterpolator Interpolator type supporting Evaluate(point)
 * @param transformPoint Optional transformation function for each sampled point
 * @returns torch::Tensor with shape (D,H,W) for 3D or (H,W) for 2D
 */
namespace ImpactTensorUtils
{
template <typename TImage, typename TInterpolator>
torch::Tensor
ImageToTensor(typename TImage::ConstPointer                                                         image,
              typename TInterpolator::Pointer                                                       interpolator,
              const std::vector<float> &                                                            voxelSize,
              const std::function<typename TImage::PointType(const typename TImage::PointType &)> & transformPoint)
{
  constexpr unsigned int Dimension = TImage::ImageDimension;
  // Compute the resampled image size based on target voxel spacing
  typename TImage::SizeType oldSize = image->GetLargestPossibleRegion().GetSize();
  typename TImage::SizeType newSize;
  for (unsigned int i = 0; i < Dimension; ++i)
    newSize[i] = static_cast<int>(oldSize[i] * image->GetSpacing()[i] / voxelSize[i] + 0.5);

  // Allocate buffer to hold interpolated intensity values
  std::vector<float> fixedImagesPatchValues;
  if (Dimension == 2)
    fixedImagesPatchValues.resize(newSize[0] * newSize[1], 0.0f);
  else
    fixedImagesPatchValues.resize(newSize[0] * newSize[1] * newSize[2], 0.0f);

  // For each target voxel, compute physical coordinates and interpolate intensity
  const itk::ZeroBasedIndexRange<Dimension> indexRange(newSize);
  unsigned int                              index = 0;
  typename TImage::PointType                imagePoint;
  for (const auto & itkIndex : indexRange)
  {
    for (unsigned int d = 0; d < Dimension; ++d)
      imagePoint[d] = image->GetOrigin()[d] + itkIndex[d] * voxelSize[d];
    fixedImagesPatchValues[index++] = interpolator->Evaluate(transformPoint ? transformPoint(imagePoint) : imagePoint);
  }
  // Wrap raw buffer into a torch tensor and clone to detach from underlying data
  if (Dimension == 2)
    return torch::from_blob(fixedImagesPatchValues.data(),
                            { static_cast<int>(newSize[0]), static_cast<int>(newSize[1]) },
                            torch::kFloat32)
      .clone();
  else
    return torch::from_blob(
             fixedImagesPatchValues.data(),
             { static_cast<int>(newSize[2]), static_cast<int>(newSize[1]), static_cast<int>(newSize[0]) },
             torch::kFloat32)
      .clone();
} // end ImageToTensor

/**
 * TensorToImage: Maps torch tensor back to ITK vector image
 *
 * @tparam TImage Reference image type for geometry info
 * @tparam TFeatureImage Output vector image type (typically itk::VectorImage)
 * @param layers Input tensor shape (C,D,H,W) or (C,H,W)
 * @returns ITK image preserving spatial properties with C-channel vectors
 */
template <typename TImage, typename TFeatureImage>
typename TFeatureImage::Pointer
TensorToImage(typename TImage::ConstPointer image, torch::Tensor layers)
{
  constexpr unsigned int Dimension = TImage::ImageDimension;
  // Rearrange tensor dimensions to match ITK vector image layout
  if (Dimension == 2)
    layers = layers.permute({ 1, 2, 0 }).contiguous().to(torch::kFloat32);
  else
    layers = layers.permute({ 1, 2, 3, 0 }).contiguous().to(torch::kFloat32);

  const unsigned int numberOfChannels = layers.size(Dimension);

  typename TFeatureImage::Pointer    itkImage = TFeatureImage::New();
  typename TFeatureImage::RegionType region;
  typename TFeatureImage::SizeType   size;

  itk::Point<double, Dimension>             origin;
  itk::Vector<float, Dimension>             spacing;
  itk::Matrix<double, Dimension, Dimension> direction;

  for (int s = 0; s < Dimension; ++s)
    size[s] = layers.size(Dimension - 1 - s);

  region.SetSize(size);
  itkImage->SetRegions(region);
  itkImage->SetVectorLength(numberOfChannels);

  auto oldSize = image->GetLargestPossibleRegion().GetSize();
  for (int i = 0; i < Dimension; ++i)
  {
    origin[i] = image->GetOrigin()[i];
    spacing[i] = oldSize[i] * image->GetSpacing()[i] / size[i];
  }
  for (int i = 0; i < Dimension; ++i)
  {
    for (int j = 0; j < Dimension; ++j)
    {
      direction[i][j] = image->GetDirection()[i][j];
    }
  }
  // Copy spatial metadata from input image to preserve geometry
  itkImage->SetOrigin(origin);
  itkImage->SetSpacing(spacing);
  itkImage->SetDirection(direction);
  itkImage->Allocate();

  const float * layersData = layers.data_ptr<float>();

  // Write each pixel vector from the tensor into the ITK vector image format
  itk::VariableLengthVector<float>          variableLengthVector(numberOfChannels);
  const itk::ZeroBasedIndexRange<Dimension> indexRange(size);
  unsigned int                              index = 0;
  for (const auto & itkIndex : indexRange)
  {
    const float * pixelPtr = layersData + (index++ * numberOfChannels);
    for (unsigned int i = 0; i < numberOfChannels; ++i)
      variableLengthVector[i] = pixelPtr[i];
    itkImage->SetPixel(itkIndex, variableLengthVector);
  }
  return itkImage;
} // end TensorToImage

/**
 * generateCartesianProduct: Computes n-dimensional Cartesian product of index sets
 *
 * @param startIndex Vector of 1D index arrays to combine
 * @param current Working array for current combination
 * @param depth Current recursion depth
 * @param result Output vector storing all index combinations
 *
 * Recursively builds all possible combinations by selecting one value from each input array
 * For N input arrays of lengths L1,L2,...,LN, generates L1*L2*...*LN total combinations
 */
inline void
generateCartesianProduct(const std::vector<std::vector<int>> & startIndex,
                         std::vector<int> &                    current,
                         unsigned int                          depth,
                         std::vector<std::vector<int>> &       result)
{
  // Recursive function to compute the Cartesian product of multiple 1D index sets
  if (depth == startIndex.size())
  {
    result.push_back(current);
    return;
  }

  for (unsigned int val : startIndex[depth])
  {
    current[depth] = val;
    generateCartesianProduct(startIndex, current, depth + 1, result);
  }
} // end generateCartesianProduct

/**
 * getPatch: Extracts and pads image patch from input tensor
 *
 * @param slice Starting coordinates for patch extraction
 * @param patchSize Desired output patch dimensions
 * @param input Source tensor to extract patch from
 * @returns Tensor containing extracted and padded patch
 *
 * Handles both 2D and 3D input tensors
 * Zero-pads extracted patch if smaller than patchSize
 */
inline torch::Tensor
getPatch(std::vector<int> slice, std::vector<int64_t> patchSize, torch::Tensor input)
{
  torch::Tensor                          patch;
  std::vector<int64_t>                   padding;
  std::vector<at::indexing::TensorIndex> indices = {
    torch::indexing::Slice(static_cast<int>(slice[0]), static_cast<int>(slice[0] + patchSize[0])),
    torch::indexing::Slice(static_cast<int>(slice[1]), static_cast<int>(slice[1] + patchSize[1]))
  };

  if (input.dim() == 3)
  {
    indices.push_back(torch::indexing::Slice(static_cast<int>(slice[2]), static_cast<int>(slice[2] + patchSize[2])));
  }
  patch = input.index(indices);

  // Pad the patch if it's smaller than the expected size
  for (int i = input.dim() - 1; i >= 0; i--)
  {
    padding.push_back(0);
    padding.push_back(patchSize[i] - patch.size(i));
  }
  return torch::constant_pad_nd(patch, padding, 0);
} // end getPatch

/**
 * pcaFit: Computes top principal components for dimensionality reduction
 *
 * @param input Tensor of shape (C,N) where C=channels, N=flattened spatial dims
 * @param new_C Target number of components to keep
 * @returns Principal component matrix shape (C,new_C) in descending eigenvalue order
 *
 * Note: Centers data and uses SVD for numerical stability
 */
inline torch::Tensor
pcaFit(torch::Tensor input, int new_C)
{

  int     C = input.size(0);
  int64_t N = std::accumulate(input.sizes().begin() + 1, input.sizes().end(), 1LL, std::multiplies<int64_t>());

  // Flatten spatial dimensions to compute PCA across feature channels
  torch::Tensor reshaped = input.view({ C, N });
  torch::Tensor centered = reshaped - reshaped.mean(1, true);
  // Center data and compute channel-wise covariance matrix
  torch::Tensor covariance = torch::matmul(centered, centered.t()) / (N - 1);

  torch::Tensor eigenvalues, eigenvectors;
  std::tie(eigenvalues, eigenvectors) = torch::linalg_eigh(covariance);
  // Select top-k eigenvectors as principal components
  return eigenvectors.narrow(1, C - new_C, new_C);
} // end pcaFit

/**
 * pcaTransform: Project data onto principal component basis
 *
 * @param input Tensor of shape (C,D,H,W) or (C,H,W) to transform
 * @param principalComponents Principal component matrix from pcaFit
 * @returns Transformed tensor with reduced channels, preserving spatial dims
 *
 * Implementation:
 * 1. Reshapes input to (C,N) where N=prod(spatial_dims)
 * 2. Centers data by channel mean
 * 3. Projects using principal component matrix
 * 4. Reshapes back to original spatial dimensions
 */
inline torch::Tensor
pcaTransform(torch::Tensor input, torch::Tensor principalComponents)
{
  int           C = input.size(0);
  int64_t       N = std::accumulate(input.sizes().begin() + 1, input.sizes().end(), 1LL, std::multiplies<int64_t>());
  torch::Tensor reshaped = input.view({ C, N });
  torch::Tensor projected = torch::matmul(principalComponents.t(), reshaped - reshaped.mean(1, true));

  std::vector<int64_t> finalShape = { principalComponents.size(1) };
  finalShape.insert(finalShape.end(), input.sizes().begin() + 1, input.sizes().end());
  return projected.view(finalShape);
} // end pcaTransform

/**
 * GetFeaturesMaps: Extract deep features from image using configured models
 *
 * @tparam TImage Input image type
 * @tparam FeaturesMaps Output feature maps container type
 * @tparam InterpolatorType Interpolator for image sampling
 * @tparam FeaturesImageType Output feature image type
 * @param image Input image to extract features from
 * @param interpolator Interpolator instance for sampling
 * @param modelsConfiguration List of deep model configurations
 * @param device Computation device (CPU/GPU)
 * @param pca Dimensions for PCA reduction per layer
 * @param principalComponents PCA matrices for dimensionality reduction
 * @param writeInputImage Optional callback to save input patches
 * @param transformPoint Optional point transformation
 * @returns Vector of feature maps, one per selected model layer
 *
 * Processing workflow:
 * 1. Converts input to tensor with proper spacing
 * 2. Extracts patches according to model config
 * 3. Processes patches through models
 * 4. Optionally reduces dimensionality via PCA
 * 5. Converts results back to ITK images
 *
 * Handles both 2D and 3D inputs with proper dimension management
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
  const std::function<typename TImage::PointType(const typename TImage::PointType &)> &            transformPoint)
{
  std::vector<FeaturesMaps> featuresMaps;
  {
    torch::NoGradGuard noGrad;
    for (const auto & config : modelsConfiguration)
    {
      // Convert image to tensor representation for deep feature extraction
      torch::Tensor inputTensor =
        ImageToTensor<TImage, InterpolatorType>(image, interpolator, config.GetVoxelSize(), transformPoint)
          .to(config.GetDataType());

      if (writeInputImage)
      {
        std::string result;

        for (int i = 0; i < config.GetVoxelSize().size(); ++i)
        {
          if (i > 0)
            result += "_";

          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << config.GetVoxelSize()[i];
          result += oss.str();
        }
        writeInputImage(image, inputTensor, result + "mm");
      }

      std::vector<int64_t> channelRepeat(config.GetDimension() + 1, 1);
      channelRepeat[0] = config.GetNumberOfChannels();

      std::vector<std::vector<int>> inputStartIndices(config.GetDimension());
      std::vector<int64_t>          patchSize = config.GetPatchSize();
      for (unsigned int dim = 0; dim < config.GetDimension(); ++dim)
      {
        if (config.GetPatchSize()[dim] <= 0)
        {
          patchSize[dim] = inputTensor.size(inputTensor.dim() - config.GetDimension() + dim);
        }
        for (int step = 0; step < std::ceil(inputTensor.size(inputTensor.dim() - config.GetDimension() + dim) /
                                            static_cast<float>(patchSize[dim]));
             ++step)
        {
          inputStartIndices[dim].push_back(patchSize[dim] * step);
        }
      }

      std::vector<std::vector<int>> inputSlices;
      std::vector<int>              inputCurrent(config.GetDimension());
      generateCartesianProduct(inputStartIndices, inputCurrent, 0, inputSlices);
      std::vector<std::vector<std::vector<int>>>             layersSlices;
      std::vector<torch::Tensor>                             layers;
      std::vector<std::vector<torch::indexing::TensorIndex>> cutting;
      if (config.GetDimension() < inputTensor.dim())
      {
        for (int64_t depthIndex = 0; depthIndex < inputTensor.size(0); ++depthIndex)
        {
          for (int sliceIndex = 0; sliceIndex < inputSlices.size(); ++sliceIndex)
          {
            torch::Tensor inputPatch = getPatch(inputSlices[sliceIndex], patchSize, inputTensor[depthIndex])
                                         .unsqueeze(0)
                                         .repeat({ torch::IntArrayRef(channelRepeat) })
                                         .unsqueeze(0)
                                         .to(device);
            std::vector<torch::jit::IValue> outputsPatch = config.GetModel().forward({ inputPatch }).toList().vec();


            if (config.GetLayersMask().size() != outputsPatch.size())
            {
              itkGenericExceptionMacro("Mismatch between layersMask size and model output layers.");
            }
            for (int layerIndex = 0, realLayerIndex = 0; layerIndex < outputsPatch.size(); ++layerIndex)
            {
              if (config.GetLayersMask()[layerIndex])
              {
                torch::Tensor layerPatch = outputsPatch[layerIndex].toTensor().squeeze(0).to(torch::kCPU);

                if (sliceIndex == 0 && depthIndex == 0)
                {
                  std::vector<torch::indexing::TensorIndex> cuttingLoc;
                  cuttingLoc.push_back(torch::indexing::Slice());
                  cuttingLoc.push_back(torch::indexing::Slice());

                  for (int r = 0; r < patchSize.size(); ++r)
                  {
                    cuttingLoc.push_back(torch::indexing::Slice(
                      0, layerPatch.size(r + 1) / static_cast<double>(patchSize[r]) * inputTensor.size(r + 1)));
                  }

                  cutting.push_back(cuttingLoc);


                  std::vector<std::vector<int>> layerStartIndices(config.GetDimension());
                  std::vector<int64_t>          layerSize(config.GetDimension() + 2);
                  layerSize[0] = layerPatch.size(0);
                  layerSize[1] = inputTensor.size(0);

                  for (unsigned int it1 = 0; it1 < config.GetDimension(); ++it1)
                  {
                    for (int it2 = 0; it2 < inputStartIndices[it1].size(); ++it2)
                    {
                      layerStartIndices[it1].push_back(layerPatch.size(it1 + 1) * it2);
                    }
                    layerSize[it1 + 2] = inputStartIndices[it1].size() * layerPatch.size(it1 + 1);
                  }
                  std::vector<int> layerCurrent(config.GetDimension());
                  layersSlices.push_back(std::vector<std::vector<int>>());
                  generateCartesianProduct(layerStartIndices, layerCurrent, 0, layersSlices[realLayerIndex]);

                  layers.push_back(torch::zeros({ torch::IntArrayRef(layerSize) }, config.GetDataType()));
                }
                layers[realLayerIndex].index_put_(
                  { torch::indexing::Slice(),
                    depthIndex,
                    torch::indexing::Slice(
                      static_cast<int>(layersSlices[realLayerIndex][sliceIndex][0]),
                      static_cast<int>(layersSlices[realLayerIndex][sliceIndex][0] + layerPatch.size(1))),
                    torch::indexing::Slice(
                      static_cast<int>(layersSlices[realLayerIndex][sliceIndex][1]),
                      static_cast<int>(layersSlices[realLayerIndex][sliceIndex][1] + layerPatch.size(2))) },
                  layerPatch);

                realLayerIndex++;
              }
            }
          }
        }
      }
      else
      {
        for (int sliceIndex = 0; sliceIndex < inputSlices.size(); ++sliceIndex)
        {
          torch::Tensor inputPatch = getPatch(inputSlices[sliceIndex], patchSize, inputTensor)
                                       .unsqueeze(0)
                                       .repeat({ torch::IntArrayRef(channelRepeat) })
                                       .unsqueeze(0)
                                       .to(device);

          std::vector<torch::jit::IValue> outputsPatch = config.GetModel().forward({ inputPatch }).toList().vec();

          if (config.GetLayersMask().size() != outputsPatch.size())
          {
            itkGenericExceptionMacro("Mismatch between layersMask size and model output layers.");
          }
          for (int layerIndex = 0, realLayerIndex = 0; layerIndex < outputsPatch.size(); ++layerIndex)
          {
            if (config.GetLayersMask()[layerIndex])
            {
              torch::Tensor layerPatch = outputsPatch[layerIndex].toTensor().squeeze(0).to(torch::kCPU);

              if (sliceIndex == 0)
              {
                std::vector<torch::indexing::TensorIndex> cuttingLoc;
                cuttingLoc.push_back(torch::indexing::Slice());
                for (int r = 0; r < patchSize.size(); ++r)
                {
                  cuttingLoc.push_back(torch::indexing::Slice(
                    0, layerPatch.size(r + 1) / static_cast<double>(patchSize[r]) * inputTensor.size(r)));
                }

                cutting.push_back(cuttingLoc);


                std::vector<std::vector<int>> layerStartIndices(config.GetDimension());
                std::vector<int64_t>          layerSize(config.GetDimension() + 1);
                layerSize[0] = layerPatch.size(0);
                for (unsigned int it1 = 0; it1 < config.GetDimension(); ++it1)
                {
                  for (int it2 = 0; it2 < inputStartIndices[it1].size(); ++it2)
                  {
                    layerStartIndices[it1].push_back(layerPatch.size(it1 + 1) * it2);
                  }
                  layerSize[it1 + 1] = inputStartIndices[it1].size() * layerPatch.size(it1 + 1);
                }
                std::vector<int> layerCurrent(config.GetDimension());
                layersSlices.push_back(std::vector<std::vector<int>>());
                generateCartesianProduct(layerStartIndices, layerCurrent, 0, layersSlices[realLayerIndex]);

                layers.push_back(torch::zeros({ torch::IntArrayRef(layerSize) }, config.GetDataType()));
              }
              const auto & slice = layersSlices[realLayerIndex][sliceIndex];


              std::vector<at::indexing::TensorIndex> slices = {
                torch::indexing::Slice(), // batch/channel dimension
                torch::indexing::Slice(static_cast<int>(slice[0]), static_cast<int>(slice[0] + layerPatch.size(1))),
                torch::indexing::Slice(static_cast<int>(slice[1]), static_cast<int>(slice[1] + layerPatch.size(2)))
              };
              if (layerPatch.dim() == 4)
              {
                slices.push_back(
                  torch::indexing::Slice(static_cast<int>(slice[2]), static_cast<int>(slice[2] + layerPatch.size(3))));
              }
              layers[realLayerIndex].index_put_(slices, layerPatch);
              realLayerIndex++;
            }
          }
        }
      }
      unsigned int a = 0;
      for (int i = 0; i < layers.size(); ++i)
      {
        torch::Tensor result = layers[i].index(cutting[i]);
        if (pca[i] > 0)
        {
          if (principalComponents.size() <= a)
          {
            principalComponents.emplace_back(pcaFit(result, pca[i]));
          }
          result = pcaTransform(result, principalComponents[a]);
          a++;
        }
        featuresMaps.emplace_back(TensorToImage<TImage, FeaturesImageType>(image, result));
      }
    }
  }
  return featuresMaps;
} // end GetFeaturesMaps


/**
 * GetModelOutputsExample: Validate model configurations with dummy inputs
 *
 * @param modelsConfig Vector of model configurations to validate
 * @param modelType String identifier for error reporting
 * @param device Computation device (CPU/GPU)
 * @returns Vector of example output tensors from each model
 *
 * Validation steps:
 * 1. Creates zero-filled dummy patches matching config specs
 * 2. Runs patches through models to verify layer structure
 * 3. Verifies layer mask compatibility
 * 4. Computes center indices for feature extraction
 *
 * Error handling:
 * - Validates dimension/channel compatibility
 * - Checks layer mask alignment with outputs
 * - Reports detailed configuration issues
 *
 * Note: Uses no_grad mode for efficiency
 */
inline std::vector<torch::Tensor>
GetModelOutputsExample(std::vector<itk::ImpactModelConfiguration> & modelsConfig,
                       const std::string &                          modelType,
                       torch::Device                                device)
{

  // For each model, create dummy patch and get output layers to check structure
  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard noGrad;
    for (int i = 0; i < modelsConfig.size(); ++i)
    {
      const auto &         config = modelsConfig[i];
      std::vector<int64_t> resizeVector(config.GetPatchSize().size() + 1, 1);
      resizeVector[0] = config.GetNumberOfChannels();
      std::vector<torch::jit::IValue> outputsList;
      auto modelInput = torch::zeros({ torch::IntArrayRef(config.GetPatchSize()) }, config.GetDataType())
                          .unsqueeze(0)
                          .repeat({ torch::IntArrayRef(resizeVector) })
                          .unsqueeze(0)
                          .clone()
                          .to(device);
      try
      {
        outputsList = config.GetModel().forward({ modelInput }).toList().vec();
      }
      catch (const std::exception & e)
      {
        itkGenericExceptionMacro(
          "ERROR: The " << modelType << " model " << i
                        << " configuration is invalid. The dimensions, number of channels, or patch size may "
                           "not meet the requirements of the model.\n"
                           "Details:\n"
                           " - Number of channels: "
                        << config.GetNumberOfChannels()
                        << "\n"
                           " - Patch size: "
                        << config.GetPatchSize()
                        << "\n"
                           " - Dimension: "
                        << config.GetDimension()
                        << "\n"
                           "Please verify the configuration to ensure compatibility with the model. \n Exception : "
                        << e.what());
      }
      if (config.GetLayersMask().size() != outputsList.size())
      {
        itkGenericExceptionMacro("Error: The number of " << modelType << " masks (" << config.GetLayersMask().size()
                                                         << ") does not match the number of layers ("
                                                         << outputsList.size()
                                                         << "). Please ensure that the configuration is consistent.");
      }

      for (int it = 0; it < outputsList.size(); ++it)
      {
        if (config.GetLayersMask()[it])
        {
          outputsTensor.push_back(outputsList[it].toTensor().to(torch::kCPU));
        }
      }
    }
    for (itk::ImpactModelConfiguration & config : modelsConfig)
    {
      std::vector<std::vector<torch::indexing::TensorIndex>> centersIndexLayers;
      for (const torch::Tensor & tensor : outputsTensor)
      {
        std::vector<torch::indexing::TensorIndex> centersIndexLayer;
        centersIndexLayer.push_back("...");
        for (int j = 2; j < tensor.dim(); ++j)
        {
          centersIndexLayer.push_back(tensor.size(j) / 2);
        }
        centersIndexLayers.push_back(centersIndexLayer);
      }
      config.SetCentersIndexLayers(centersIndexLayers);
    }
  }
  return outputsTensor;
} // end GetModelOutputsExample

/**
 * GetPatchIndex: Generates sampling grid for patch extraction
 *
 * @param modelConfiguration Model-specific patch configuration
 * @param randomGenerator RNG for stochastic patch orientation
 * @param dimension Target space dimension
 * @returns List of sampling coordinates per patch point
 *
 * For 2D patches: Applies random rotation to sampling
 */
inline std::vector<std::vector<float>>
GetPatchIndex(const itk::ImpactModelConfiguration & modelConfiguration,
              std::mt19937 &                        randomGenerator,
              unsigned int                          dimension)
{
  if (dimension == modelConfiguration.GetPatchSize().size())
  {
    return modelConfiguration.GetPatchIndex();
  }
  else
  {

    using MatrixType = itk::Matrix<float, 3, 3>;
    using Point3D = itk::Point<float, 3>;
    std::uniform_real_distribution<double> angleDist(0.0, 2.0 * M_PI);

    double radX = angleDist(randomGenerator);
    double radY = angleDist(randomGenerator);
    double radZ = angleDist(randomGenerator);

    MatrixType rotationX;
    MatrixType rotationY;
    MatrixType rotationZ;

    rotationX.SetIdentity();
    rotationY.SetIdentity();
    rotationZ.SetIdentity();

    rotationX[1][1] = cos(radX);
    rotationX[1][2] = -sin(radX);
    rotationX[2][1] = sin(radX);
    rotationX[2][2] = cos(radX);

    rotationY[0][0] = cos(radY);
    rotationY[0][2] = sin(radY);
    rotationY[2][0] = -sin(radY);
    rotationY[2][2] = cos(radY);

    rotationZ[0][0] = cos(radZ);
    rotationZ[0][1] = -sin(radZ);
    rotationZ[1][0] = sin(radZ);
    rotationZ[1][1] = cos(radZ);

    MatrixType                      matrix = rotationZ * rotationY * rotationX;
    std::vector<std::vector<float>> patchIndex;

    for (int y = 0; y < modelConfiguration.GetPatchSize()[1]; ++y)
    {
      for (int x = 0; x < modelConfiguration.GetPatchSize()[0]; ++x)
      {
        Point3D point({ (x - modelConfiguration.GetPatchSize()[0] / 2) * modelConfiguration.GetVoxelSize()[0],
                        (y - modelConfiguration.GetPatchSize()[1] / 2) * modelConfiguration.GetVoxelSize()[1],
                        0 });
        point = matrix * point;
        std::vector<float> vec(3);
        vec[0] = point[0];
        vec[1] = point[1];
        vec[2] = point[2];
        patchIndex.push_back(vec);
      }
    }
    return patchIndex;
  }
} // end GetPatchIndex

/**
 * GenerateOutputs: Batch processing of image patches through deep models
 *
 * @param modelConfig Configuration per model including architecture and layers
 * @param fixedPoints List of control points to extract patches around
 * @param patchIndex Pre-computed sampling indices for each patch
 * @param subsetsOfFeatures Selected feature channels per layer
 * @returns List of output tensors, one per selected layer per model
 *
 * Performance note: Processes patches in batches to maximize GPU utilization
 */
template <typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputs(const std::vector<itk::ImpactModelConfiguration> &                    modelConfig,
                const std::vector<ImagePointType> &                                   fixedPoints,
                const std::vector<std::vector<std::vector<std::vector<float>>>> &     patchIndex,
                const std::vector<torch::Tensor>                                      subsetsOfFeatures,
                torch::Device                                                         device,
                const ImpactTensorUtils::ImagesPatchValuesEvaluator<ImagePointType> & imagesPatchValuesEvaluator)
{

  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard noGrad;
    unsigned int       nbSample = fixedPoints.size();

    int a = 0;
    for (int i = 0; i < modelConfig.size(); ++i)
    {
      const auto & config = modelConfig[i];

      std::vector<int64_t> sizes(config.GetPatchSize().size() + 1, -1);
      sizes[0] = nbSample;

      torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(config.GetPatchSize()) }, config.GetDataType())
                                         .unsqueeze(0)
                                         .expand(sizes)
                                         .unsqueeze(1)
                                         .clone();

      for (unsigned int s = 0; s < nbSample; ++s)
      {
        patchValueTensor[s] =
          imagesPatchValuesEvaluator(fixedPoints[s], patchIndex[i][s], config.GetPatchSize()).to(config.GetDataType());
      }

      std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
      resizeVector[1] = config.GetNumberOfChannels();
      std::vector<torch::jit::IValue> outputsList =
        config.GetModel()
          .forward({ patchValueTensor.to(device).repeat({ torch::IntArrayRef(resizeVector) }).clone() })
          .toList()
          .vec();

      for (int it = 0; it < outputsList.size(); ++it)
      {
        if (config.GetLayersMask()[it])
        {
          outputsTensor.push_back(outputsList[it]
                                    .toTensor()
                                    .index(config.GetCentersIndexLayers()[a])
                                    .index_select(1, subsetsOfFeatures[a])
                                    .to(torch::kFloat32));
          a++;
        }
      }
    }
  }
  return outputsTensor;
} // end GenerateOutputs

/**
 * GenerateOutputsAndJacobian: Computes model outputs and their Jacobians
 *
 * @param modelConfig List of model configurations with architectures
 * @param fixedPoints Control points for patch extraction
 * @param patchIndex Sampling grid coordinates per patch
 * @param subsetsOfFeatures Feature channel selections
 * @param fixedOutputsTensor Reference outputs for loss calculation
 * @param device Computation device (CPU/GPU)
 * @param losses Loss function objects per output layer
 * @param imagesPatchValuesAndJacobiansEvaluator Callback for patch and gradient evaluation
 * @returns List of Jacobian tensors for each model output
 *
 * Performance note: Batches computation and uses autograd for efficiency
 * Handles multiple models and multiple output layers per model
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
                             imagesPatchValuesAndJacobiansEvaluator)
{
  std::vector<torch::Tensor> layersJacobian;

  unsigned int nbSample = fixedPoints.size();
  unsigned int dimension = fixedPoints[0].size();

  int a = 0;
  for (int i = 0; i < modelConfig.size(); ++i)
  {
    const auto & config = modelConfig[i];

    std::vector<int64_t> sizes(config.GetPatchSize().size() + 1, -1);
    sizes[0] = nbSample;

    torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(config.GetPatchSize()) }, config.GetDataType())
                                       .unsqueeze(0)
                                       .expand(sizes)
                                       .unsqueeze(1)
                                       .clone();
    torch::Tensor imagesPatchesJacobians =
      torch::zeros({ nbSample, static_cast<int64_t>(patchIndex[i][0].size()), dimension }, torch::kFloat32);

    for (unsigned int s = 0; s < nbSample; ++s)
    {
      patchValueTensor[s] = imagesPatchValuesAndJacobiansEvaluator(
                              fixedPoints[s], imagesPatchesJacobians, patchIndex[i][s], config.GetPatchSize(), s)
                              .to(config.GetDataType());
    }


    std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
    resizeVector[1] = config.GetNumberOfChannels();
    patchValueTensor =
      patchValueTensor.to(device).repeat({ torch::IntArrayRef(resizeVector) }).clone().set_requires_grad(true);
    imagesPatchesJacobians = imagesPatchesJacobians.to(device).repeat({ 1, config.GetNumberOfChannels(), 1 }).clone();

    std::vector<torch::jit::IValue> outputsList = config.GetModel().forward({ patchValueTensor }).toList().vec();
    torch::Tensor                   layer, diffLayer, modelJacobian;
    for (int it = 0; it < outputsList.size(); ++it)
    {
      if (config.GetLayersMask()[it])
      {
        int nb = std::accumulate(config.GetLayersMask().begin(), config.GetLayersMask().end(), 0);

        layer = outputsList[it]
                  .toTensor()
                  .index(config.GetCentersIndexLayers()[a])
                  .index_select(1, subsetsOfFeatures[a])
                  .to(torch::kFloat32);
        torch::Tensor gradientModulator = losses[a]->updateValueAndGetGradientModulator(fixedOutputsTensor[a], layer);
        std::vector<torch::Tensor> modelJacobians;
        layersJacobian.push_back(
          torch::bmm(torch::autograd::grad({ layer }, { patchValueTensor }, { gradientModulator }, nb > 1, false)[0]
                       .flatten(1)
                       .unsqueeze(1)
                       .to(torch::kFloat32),
                     imagesPatchesJacobians));

        a++;
      }
    }
  }
  return layersJacobian;
} // end GenerateOutputsAndJacobian


} // namespace ImpactTensorUtils

#endif // end #ifndef _ImpactTensorUtils_hxx
