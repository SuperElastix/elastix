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
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "elxlog.h"
#include <ATen/autocast_mode.h>

/**
 * ******************* ImageToTensor ***********************
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
  auto                 oldSize = image->GetLargestPossibleRegion().GetSize();
  std::vector<int64_t> newSize(Dimension);
  for (unsigned int i = 0; i < Dimension; ++i)
  {
    newSize[i] = static_cast<int>(oldSize[i] * image->GetSpacing()[i] / voxelSize[i] + 0.5);
  }
  // Allocate buffer to hold interpolated intensity values
  std::vector<float> fixedImagesPatchValues;


  if (Dimension == 2)
  {
    fixedImagesPatchValues.resize(newSize[0] * newSize[1], 0.0f);
// For each target voxel, compute physical coordinates and interpolate intensity
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < newSize[1]; ++y)
    {
      for (int x = 0; x < newSize[0]; ++x)
      {
        unsigned int               index = y * newSize[0] + x;
        typename TImage::PointType imagePoint;
        imagePoint[0] = image->GetOrigin()[0] + x * voxelSize[0];
        imagePoint[1] = image->GetOrigin()[1] + y * voxelSize[1];
        if (transformPoint)
        {
          fixedImagesPatchValues[index] = interpolator->Evaluate(transformPoint(imagePoint));
        }
        else
        {
          fixedImagesPatchValues[index] = interpolator->Evaluate(imagePoint);
        }
      }
    }
    // Wrap raw buffer into a torch tensor and clone to detach from underlying data
    return torch::from_blob(fixedImagesPatchValues.data(), { newSize[0], newSize[1] }, torch::kFloat).clone();
  }
  else
  {
    fixedImagesPatchValues.resize(newSize[0] * newSize[1] * newSize[2], 0.0f);
// For each target voxel, compute physical coordinates and interpolate intensity
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int z = 0; z < newSize[2]; ++z)
    {
      for (int y = 0; y < newSize[1]; ++y)
      {
        for (int x = 0; x < newSize[0]; ++x)
        {
          unsigned int               index = z * newSize[1] * newSize[0] + y * newSize[0] + x;
          typename TImage::PointType imagePoint;
          imagePoint[0] = image->GetOrigin()[0] + x * voxelSize[0];
          imagePoint[1] = image->GetOrigin()[1] + y * voxelSize[1];
          imagePoint[2] = image->GetOrigin()[2] + z * voxelSize[2];
          if (transformPoint)
          {
            fixedImagesPatchValues[index] = interpolator->Evaluate(transformPoint(imagePoint));
          }
          else
          {
            fixedImagesPatchValues[index] = interpolator->Evaluate(imagePoint);
          }
        }
      }
    }
    // Wrap raw buffer into a torch tensor and clone to detach from underlying data
    return torch::from_blob(fixedImagesPatchValues.data(), { newSize[2], newSize[1], newSize[0] }, torch::kFloat)
      .clone();
  }
} // end ImageToTensor

/**
 * ******************* TensorToImage ***********************
 */
template <typename TImage, typename TFeatureImage>
typename TFeatureImage::Pointer
TensorToImage(typename TImage::ConstPointer image, torch::Tensor layers)
{
  constexpr unsigned int Dimension = TImage::ImageDimension;
  // Rearrange tensor dimensions to match ITK vector image layout
  if (Dimension == 2)
  {
    layers = layers.permute({ 1, 2, 0 }).contiguous();
    ;
  }
  else
  {
    layers = layers.permute({ 1, 2, 3, 0 }).contiguous();
    ;
  }

  const unsigned int numberOfChannels = layers.size(Dimension);

  typename TFeatureImage::Pointer    itkImage = TFeatureImage::New();
  typename TFeatureImage::RegionType region;
  typename TFeatureImage::SizeType   size;

  itk::Point<double, Dimension>             origin;
  itk::Vector<float, Dimension>             spacing;
  itk::Matrix<double, Dimension, Dimension> direction;

  for (int s = 0; s < Dimension; s++)
  {
    size[s] = layers.size(Dimension - 1 - s);
  }
  region.SetSize(size);
  itkImage->SetRegions(region);
  itkImage->SetVectorLength(numberOfChannels);

  auto oldSize = image->GetLargestPossibleRegion().GetSize();
  for (int i = 0; i < Dimension; i++)
  {
    origin[i] = image->GetOrigin()[i];
    spacing[i] = oldSize[i] * image->GetSpacing()[i] / size[i];
  }
  for (unsigned int i = 0; i < Dimension; ++i)
  {
    for (unsigned int j = 0; j < Dimension; ++j)
    {
      direction[i][j] = image->GetDirection()[i][j];
    }
  }
  // Copy spatial metadata from input image to preserve geometry
  itkImage->SetOrigin(origin);
  itkImage->SetSpacing(spacing);
  itkImage->SetDirection(direction);
  itkImage->Allocate();

  const float *      layersData = layers.data_ptr<float>();
  const unsigned int rowStride = size[0] * numberOfChannels;
  const unsigned int sliceStride = size[0] * size[1] * numberOfChannels;

  // Write each pixel vector from the tensor into the ITK vector image format
  if (Dimension == 2)
  {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int x = 0; x < size[1]; ++x)
    {
      for (int y = 0; y < size[0]; ++y)
      {
        const float *                    pixelPtr = layersData + x * sliceStride + y * rowStride;
        itk::VariableLengthVector<float> variableLengthVector(numberOfChannels);
        for (unsigned int i = 0; i < numberOfChannels; i++)
        {
          variableLengthVector[i] = pixelPtr[i];
        }
        typename TFeatureImage::IndexType index;
        index[0] = y;
        index[1] = x;
        itkImage->SetPixel(index, variableLengthVector);
      }
    }
  }
  else
  {
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int x = 0; x < size[2]; ++x)
    {
      for (int y = 0; y < size[1]; ++y)
      {
        for (int z = 0; z < size[0]; ++z)
        {
          const float * pixelPtr = layersData + x * sliceStride + y * rowStride + z * numberOfChannels;
          itk::VariableLengthVector<float> variableLengthVector(numberOfChannels);
          for (unsigned int i = 0; i < numberOfChannels; i++)
          {
            variableLengthVector[i] = pixelPtr[i]; // layers[x][y][z][i].item<float>();
          }
          typename TFeatureImage::IndexType index;
          index[0] = z;
          index[1] = y;
          index[2] = x;
          itkImage->SetPixel(index, variableLengthVector);
        }
      }
    }
  }
  return itkImage;
} // end TensorToImage

/**
 * ******************* generateCartesianProduct ***********************
 */
void
generateCartesianProduct(const std::vector<std::vector<int>> & startIndex,
                         std::vector<int> &                    current,
                         size_t                                depth,
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
 * ******************* getPatch ***********************
 */
torch::Tensor
getPatch(std::vector<int> slice, std::vector<long> patchSize, torch::Tensor input)
{
  torch::Tensor        patch;
  std::vector<int64_t> padding;
  if (input.dim() == 2)
  {
    patch =
      input.index({ torch::indexing::Slice(static_cast<int>(slice[0]), static_cast<int>(slice[0] + patchSize[0])),
                    torch::indexing::Slice(static_cast<int>(slice[1]), static_cast<int>(slice[1] + patchSize[1])) });
  }
  else
  {
    patch =
      input.index({ torch::indexing::Slice(static_cast<int>(slice[0]), static_cast<int>(slice[0] + patchSize[0])),
                    torch::indexing::Slice(static_cast<int>(slice[1]), static_cast<int>(slice[1] + patchSize[1])),
                    torch::indexing::Slice(static_cast<int>(slice[2]), static_cast<int>(slice[2] + patchSize[2])) });
  }
  // Pad the patch if it's smaller than the expected size
  for (int i = input.dim() - 1; i >= 0; i--)
  {
    padding.push_back(0);
    padding.push_back(patchSize[i] - patch.size(i));
  }
  return torch::constant_pad_nd(patch, padding, 0);
} // end getPatch

/**
 * ******************* pca_fit ***********************
 */
torch::Tensor
pca_fit(torch::Tensor input, int new_C)
{
  int C = input.size(0);
  int D = input.size(1);
  int H = input.size(2);
  int W = input.size(3);
  // Flatten spatial dimensions to compute PCA across feature channels
  torch::Tensor reshaped = input.view({ C, D * H * W });
  torch::Tensor centered = reshaped - reshaped.mean(1, true);
  // Center data and compute channel-wise covariance matrix
  torch::Tensor covariance = torch::matmul(centered, centered.t()) / (D * H * W - 1);

  torch::Tensor eigenvalues, eigenvectors;
  std::tie(eigenvalues, eigenvectors) = torch::linalg_eigh(covariance);
  // Select top-k eigenvectors as principal components
  return eigenvectors.narrow(1, C - new_C, new_C);
} // end pca_fit

/**
 * ******************* pca_transform ***********************
 */
torch::Tensor
pca_transform(torch::Tensor input, torch::Tensor principal_components)
{
  int           C = input.size(0);
  int           D = input.size(1);
  int           H = input.size(2);
  int           W = input.size(3);
  torch::Tensor reshaped = input.view({ C, D * H * W });
  return torch::matmul(principal_components.t(), reshaped - reshaped.mean(1, true))
    .view({ principal_components.size(1), D, H, W });
} // end pca_transform

/**
 * ******************* GetFeaturesMaps ***********************
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
  torch::Device                                                                                    gpu,
  std::vector<unsigned int>                                                                        pca,
  std::vector<torch::Tensor> &                                                                     principal_components,
  const std::function<void(typename TImage::ConstPointer, torch::Tensor &, const std::string &)> & writeInputImage,
  const std::function<typename TImage::PointType(const typename TImage::PointType &)> &            transformPoint)
{
  std::vector<FeaturesMaps> featuresMaps;
  {
    torch::NoGradGuard no_grad;
    for (const auto & config : modelsConfiguration)
    {
      // Convert image to tensor representation for deep feature extraction
      torch::Tensor inputTensor =
        ImageToTensor<TImage, InterpolatorType>(image, interpolator, config.m_voxelSize, transformPoint);
      if (writeInputImage)
      {
        std::string result;

        for (int i = 0; i < config.m_voxelSize.size(); ++i)
        {
          if (i > 0)
            result += "_";

          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << config.m_voxelSize[i];
          result += oss.str();
        }
        writeInputImage(image, inputTensor, result + "mm");
      }
      std::vector<int64_t> channelRepeat(config.m_dimension + 1, 1);
      channelRepeat[0] = config.m_numberOfChannels;

      std::vector<std::vector<int>> inputStartIndices(config.m_dimension);
      std::vector<long>             patchSize = config.m_patchSize;
      for (int dim = 0; dim < config.m_dimension; dim++)
      {
        if (config.m_patchSize[dim] <= 0)
        {
          patchSize[dim] = inputTensor.size(inputTensor.dim() - config.m_dimension + dim);
        }
        for (int step = 0; step < std::ceil(inputTensor.size(inputTensor.dim() - config.m_dimension + dim) /
                                            static_cast<float>(patchSize[dim]));
             step++)
        {
          inputStartIndices[dim].push_back(patchSize[dim] * step);
        }
      }

      std::vector<std::vector<int>> inputSlices;
      std::vector<int>              inputCurrent(config.m_dimension);
      generateCartesianProduct(inputStartIndices, inputCurrent, 0, inputSlices);
      std::vector<std::vector<std::vector<int>>> layersSlices;
      std::vector<torch::Tensor>                 layers;
      std::vector<std::vector<double>>           ratio;

      if (config.m_dimension < inputTensor.dim())
      {
        for (int depthIndex = 0; depthIndex < inputTensor.size(0); depthIndex++)
        {
          for (int sliceIndex = 0; sliceIndex < inputSlices.size(); sliceIndex++)
          {
            torch::Tensor inputPatch = getPatch(inputSlices[sliceIndex], patchSize, inputTensor[depthIndex])
                                         .unsqueeze(0)
                                         .repeat({ torch::IntArrayRef(channelRepeat) })
                                         .unsqueeze(0)
                                         .to(gpu);
            std::vector<torch::jit::IValue> outputsPatch = config.m_model->forward({ inputPatch }).toList().vec();


            if (config.m_layersMask.size() != outputsPatch.size())
            {
              itkGenericExceptionMacro("Mismatch between layersMask size and model output layers.");
            }
            for (size_t layerIndex = 0, realLayerIndex = 0; layerIndex < outputsPatch.size(); layerIndex++)
            {
              if (config.m_layersMask[layerIndex])
              {
                torch::Tensor layerPatch =
                  outputsPatch[layerIndex].toTensor().squeeze(0).to(torch::kCPU).to(torch::kFloat);

                if (sliceIndex == 0 && depthIndex == 0)
                {
                  ratio.push_back({ layerPatch.size(1) / static_cast<double>(patchSize[0]),
                                    layerPatch.size(2) / static_cast<double>(patchSize[1]) });

                  std::vector<std::vector<int>> layerStartIndices(config.m_dimension);
                  std::vector<long>             layerSize(config.m_dimension + 2);
                  layerSize[0] = layerPatch.size(0);
                  layerSize[1] = inputTensor.size(0);

                  for (int it1 = 0; it1 < config.m_dimension; it1++)
                  {
                    for (int it2 = 0; it2 < inputStartIndices[it1].size(); it2++)
                    {
                      layerStartIndices[it1].push_back(layerPatch.size(it1 + 1) * it2);
                    }
                    layerSize[it1 + 2] = inputStartIndices[it1].size() * layerPatch.size(it1 + 1);
                  }
                  std::vector<int> layerCurrent(config.m_dimension);
                  layersSlices.push_back(std::vector<std::vector<int>>());
                  generateCartesianProduct(layerStartIndices, layerCurrent, 0, layersSlices[realLayerIndex]);

                  layers.push_back(torch::zeros({ torch::IntArrayRef(layerSize) }, torch::kFloat));
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
        for (int sliceIndex = 0; sliceIndex < inputSlices.size(); sliceIndex++)
        {
          torch::Tensor inputPatch = getPatch(inputSlices[sliceIndex], patchSize, inputTensor)
                                       .unsqueeze(0)
                                       .repeat({ torch::IntArrayRef(channelRepeat) })
                                       .unsqueeze(0)
                                       .to(gpu);
          std::vector<torch::jit::IValue> outputsPatch = config.m_model->forward({ inputPatch }).toList().vec();

          if (config.m_layersMask.size() != outputsPatch.size())
          {
            itkGenericExceptionMacro("Mismatch between layersMask size and model output layers.");
          }
          for (size_t layerIndex = 0, realLayerIndex = 0; layerIndex < outputsPatch.size(); layerIndex++)
          {
            if (config.m_layersMask[layerIndex])
            {
              torch::Tensor layerPatch =
                outputsPatch[layerIndex].toTensor().squeeze(0).to(torch::kCPU).to(torch::kFloat);

              if (sliceIndex == 0)
              {
                ratio.push_back({ layerPatch.size(1) / static_cast<double>(patchSize[0]),
                                  layerPatch.size(2) / static_cast<double>(patchSize[1]),
                                  layerPatch.size(3) / static_cast<double>(patchSize[2]) });

                std::vector<std::vector<int>> layerStartIndices(config.m_dimension);
                std::vector<long>             layerSize(config.m_dimension + 1);
                layerSize[0] = layerPatch.size(0);
                for (int it1 = 0; it1 < config.m_dimension; it1++)
                {
                  for (int it2 = 0; it2 < inputStartIndices[it1].size(); it2++)
                  {
                    layerStartIndices[it1].push_back(layerPatch.size(it1 + 1) * it2);
                  }
                  layerSize[it1 + 1] = inputStartIndices[it1].size() * layerPatch.size(it1 + 1);
                }
                std::vector<int> layerCurrent(config.m_dimension);
                layersSlices.push_back(std::vector<std::vector<int>>());
                generateCartesianProduct(layerStartIndices, layerCurrent, 0, layersSlices[realLayerIndex]);

                layers.push_back(torch::zeros({ torch::IntArrayRef(layerSize) }, torch::kFloat));
              }
              layers[realLayerIndex].index_put_(
                { torch::indexing::Slice(),
                  torch::indexing::Slice(
                    static_cast<int>(layersSlices[realLayerIndex][sliceIndex][0]),
                    static_cast<int>(layersSlices[realLayerIndex][sliceIndex][0] + layerPatch.size(1))),
                  torch::indexing::Slice(
                    static_cast<int>(layersSlices[realLayerIndex][sliceIndex][1]),
                    static_cast<int>(layersSlices[realLayerIndex][sliceIndex][1] + layerPatch.size(2))),
                  torch::indexing::Slice(
                    static_cast<int>(layersSlices[realLayerIndex][sliceIndex][2]),
                    static_cast<int>(layersSlices[realLayerIndex][sliceIndex][2] + layerPatch.size(3))) },
                layerPatch);

              realLayerIndex++;
            }
          }
        }
      }
      unsigned int a = 0;
      for (size_t i = 0; i < layers.size(); i++)
      {
        std::vector<torch::indexing::TensorIndex> cutting;
        int                                       j = 0;
        for (; j < layers[i].dim() - ratio[i].size(); j++)
        {
          cutting.push_back(torch::indexing::Slice());
        }
        for (int r = 0; r < ratio[i].size(); r++)
        {
          int end = -layers[i].size(j + r) + ratio[i][r] * inputTensor.size(j - 1 + r);
          if (end <= 0)
          {
            end = layers[i].size(j + r);
          }
          cutting.push_back(torch::indexing::Slice(0, end));
        }
        torch::Tensor result = layers[i].index(cutting);

        if (pca[i] > 0)
        {
          if (principal_components.size() <= a)
          {
            principal_components.emplace_back(pca_fit(result, pca[i]));
          }
          result = pca_transform(result, principal_components[a]);
          a++;
        }
        featuresMaps.emplace_back(TensorToImage<TImage, FeaturesImageType>(image, result));
      }
    }
  }
  return featuresMaps;
} // end GetFeaturesMaps


/**
 * **************** GetModelOutputsExample ****************
 */
template <typename ModelConfiguration>
std::vector<torch::Tensor>
GetModelOutputsExample(std::vector<ModelConfiguration> & modelsConfig, const std::string & modelType, torch::Device gpu)
{

  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < modelsConfig.size(); ++i)
    {
      const auto &         config = modelsConfig[i];
      std::vector<int64_t> resizeVector(config.m_patchSize.size() + 1, 1);
      resizeVector[0] = config.m_numberOfChannels;
      std::vector<torch::jit::IValue> outputsList;
      auto modelInput = torch::zeros({ torch::IntArrayRef(config.m_patchSize) }, torch::kFloat)
                          .unsqueeze(0)
                          .repeat({ torch::IntArrayRef(resizeVector) })
                          .unsqueeze(0)
                          .clone()
                          .to(gpu);
      try
      {
        outputsList = config.m_model->forward({ modelInput }).toList().vec();
      }
      catch (const std::exception & e)
      {
        itkGenericExceptionMacro("ERROR: The "
                                 << modelType << " model " << i
                                 << " configuration is invalid. The dimensions, number of channels, or patch size may "
                                    "not meet the requirements of the model.\n"
                                    "Details:\n"
                                    " - Number of channels: "
                                 << config.m_numberOfChannels
                                 << "\n"
                                    " - Patch size: "
                                 << config.m_patchSize
                                 << "\n"
                                    " - Dimension: "
                                 << config.m_dimension
                                 << "\n"
                                    "Please verify the configuration to ensure compatibility with the model.");
      }
      if (config.m_layersMask.size() != outputsList.size())
      {
        itkGenericExceptionMacro("Error: The number of " << modelType << " masks (" << config.m_layersMask.size()
                                                         << ") does not match the number of layers ("
                                                         << outputsList.size()
                                                         << "). Please ensure that the configuration is consistent.");
      }

      for (size_t it = 0; it < outputsList.size(); ++it)
      {
        if (config.m_layersMask[it])
        {
          outputsTensor.push_back(outputsList[it].toTensor().to(torch::kCPU).to(torch::kFloat));
        }
      }
    }
    for (size_t i = 0; i < modelsConfig.size(); ++i)
    {
      auto & config = modelsConfig[i];
      config.m_centersIndexLayers.clear();
      for (size_t it = 0; it < outputsTensor.size(); ++it)
      {
        std::vector<torch::indexing::TensorIndex> centersIndexLayer;
        centersIndexLayer.push_back("...");
        for (size_t j = 2; j < outputsTensor[it].dim(); ++j)
        {
          centersIndexLayer.push_back(outputsTensor[it].size(j) / 2);
        }
        config.m_centersIndexLayers.push_back(centersIndexLayer);
      }
    }
  }
  return outputsTensor;
} // end GetModelOutputsExample

/**
 * ******************* GetPatchIndex ***********************
 */
template <typename ModelConfiguration>
std::vector<std::vector<float>>
GetPatchIndex(ModelConfiguration modelConfiguration, unsigned int dimension)
{
  if (dimension == modelConfiguration.m_patchSize.size())
  {
    return modelConfiguration.m_patchIndex;
  }
  else
  {

    using MatrixType = itk::Matrix<float, 3, 3>;
    using Point3D = itk::Point<float, 3>;

    double radX = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
    double radY = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
    double radZ = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;

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

    for (int y = 0; y < modelConfiguration.m_patchSize[1]; ++y)
    {
      for (int x = 0; x < modelConfiguration.m_patchSize[0]; ++x)
      {
        Point3D point({ (x - modelConfiguration.m_patchSize[0] / 2) * modelConfiguration.m_voxelSize[0],
                        (y - modelConfiguration.m_patchSize[1] / 2) * modelConfiguration.m_voxelSize[1],
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
 * ******************* GenerateOutputs ***********************
 */
template <typename ModelConfiguration, typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputs(const std::vector<ModelConfiguration> &                                  modelConfig,
                const std::vector<ImagePointType> &                                      fixedPoints,
                const std::vector<std::vector<std::vector<std::vector<float>>>> &        patchIndex,
                const std::vector<torch::Tensor>                                         subsetsOfFeatures,
                torch::Device                                                            gpu,
                const std::function<typename torch::Tensor(const ImagePointType &,
                                                           const std::vector<std::vector<float>> &,
                                                           const std::vector<long> &)> & evaluator)
{

  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard no_grad;
    unsigned int       nbSample = fixedPoints.size();

    int a = 0;
    for (size_t i = 0; i < modelConfig.size(); ++i)
    {
      const auto & config = modelConfig[i];

      std::vector<int64_t> sizes(config.m_patchSize.size() + 1, -1);
      sizes[0] = nbSample;

      torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(config.m_patchSize) }, torch::kFloat)
                                         .unsqueeze(0)
                                         .expand(sizes)
                                         .unsqueeze(1)
                                         .clone();

      for (size_t s = 0; s < nbSample; ++s)
      {
        patchValueTensor[s] = evaluator(fixedPoints[s], patchIndex[i][s], config.m_patchSize);
      }

      std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
      resizeVector[1] = config.m_numberOfChannels;
      std::vector<torch::jit::IValue> outputsList =
        config.m_model->forward({ patchValueTensor.to(gpu).repeat({ torch::IntArrayRef(resizeVector) }).clone() })
          .toList()
          .vec();

      for (size_t it = 0; it < outputsList.size(); ++it)
      {
        if (config.m_layersMask[it])
        {
          outputsTensor.push_back(outputsList[it]
                                    .toTensor()
                                    .index(config.m_centersIndexLayers[a])
                                    .index_select(1, subsetsOfFeatures[a])
                                    .to(torch::kFloat));
          a++;
        }
      }
    }
  }
  return outputsTensor;
} // end GenerateOutputs

/**
 * ******************* GenerateOutputsAndJacobian ***********************
 */
template <typename ModelConfiguration, typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputsAndJacobian(const std::vector<ModelConfiguration> &                           modelConfig,
                           const std::vector<ImagePointType> &                               fixedPoints,
                           const std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex,
                           std::vector<torch::Tensor>                                        subsetsOfFeatures,
                           std::vector<torch::Tensor>                                        fixedOutputsTensor,
                           torch::Device                                                     gpu,
                           std::vector<std::unique_ptr<ImpactLoss::Loss>> &                  losses,
                           const std::function<typename torch::Tensor(const ImagePointType &,
                                                                      torch::Tensor &,
                                                                      const std::vector<std::vector<float>> &,
                                                                      const std::vector<long> &,
                                                                      int)> &                evaluator)
{
  std::vector<torch::Tensor> layersJacobian;

  unsigned int nbSample = fixedPoints.size();
  unsigned int dimension = fixedPoints[0].size();

  int a = 0;
  for (size_t i = 0; i < modelConfig.size(); ++i)
  {
    const auto & config = modelConfig[i];

    std::vector<int64_t> sizes(config.m_patchSize.size() + 1, -1);
    sizes[0] = nbSample;

    torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(config.m_patchSize) }, torch::kFloat)
                                       .unsqueeze(0)
                                       .expand(sizes)
                                       .unsqueeze(1)
                                       .clone();
    torch::Tensor imagesPatchesJacobians =
      torch::zeros({ nbSample, static_cast<long>(patchIndex[i][0].size()), dimension }, torch::kFloat);

    for (size_t s = 0; s < nbSample; ++s)
    {
      patchValueTensor[s] = evaluator(fixedPoints[s], imagesPatchesJacobians, patchIndex[i][s], config.m_patchSize, s);
    }


    std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
    resizeVector[1] = config.m_numberOfChannels;
    patchValueTensor =
      patchValueTensor.to(gpu).repeat({ torch::IntArrayRef(resizeVector) }).clone().set_requires_grad(true);
    imagesPatchesJacobians = imagesPatchesJacobians.to(gpu).repeat({ 1, config.m_numberOfChannels, 1 }).clone();

    std::vector<torch::jit::IValue> outputsList = config.m_model->forward({ patchValueTensor }).toList().vec();
    torch::Tensor                   layer, diffLayer, modelJacobian;
    for (size_t it = 0; it < outputsList.size(); ++it)
    {
      if (config.m_layersMask[it])
      {
        int nb = std::accumulate(config.m_layersMask.begin(), config.m_layersMask.end(), 0);

        layer = outputsList[it]
                  .toTensor()
                  .index(config.m_centersIndexLayers[a])
                  .index_select(1, subsetsOfFeatures[a])
                  .to(torch::kFloat);
        torch::Tensor gradientModulator = losses[a]->updateValueAndGetGradientModulator(fixedOutputsTensor[a], layer);
        std::vector<torch::Tensor> modelJacobians;
        layersJacobian.push_back(
          torch::bmm(torch::autograd::grad({ layer }, { patchValueTensor }, { gradientModulator }, nb > 1, false)[0]
                       .flatten(1)
                       .unsqueeze(1),
                     imagesPatchesJacobians));

        a++;
      }
    }
  }
  return layersJacobian;
} // end GenerateOutputsAndJacobian


} // namespace ImpactTensorUtils

#endif // end #ifndef _ImpactTensorUtils_hxx
