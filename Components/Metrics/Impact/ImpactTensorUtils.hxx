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
    return torch::from_blob(fixedImagesPatchValues.data(), { newSize[0], newSize[1] }, torch::kFloat32).clone();
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
    return torch::from_blob(fixedImagesPatchValues.data(), { newSize[2], newSize[1], newSize[0] }, torch::kFloat32)
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
    layers = layers.permute({ 1, 2, 0 }).contiguous().to(torch::kFloat32);
    ;
  }
  else
  {
    layers = layers.permute({ 1, 2, 3, 0 }).contiguous().to(torch::kFloat32);
    ;
  }
  const unsigned int numberOfChannels = layers.size(Dimension);

  typename TFeatureImage::Pointer    itkImage = TFeatureImage::New();
  typename TFeatureImage::RegionType region;
  typename TFeatureImage::SizeType   size;

  itk::Point<double, Dimension>             origin;
  itk::Vector<float, Dimension>             spacing;
  itk::Matrix<double, Dimension, Dimension> direction;

  for (int s = 0; s < Dimension; ++s)
  {
    size[s] = layers.size(Dimension - 1 - s);
  }
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
        const float *                    pixelPtr = layersData + x * rowStride + y * numberOfChannels;
        itk::VariableLengthVector<float> variableLengthVector(numberOfChannels);
        for (unsigned int i = 0; i < numberOfChannels; ++i)
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
          for (unsigned int i = 0; i < numberOfChannels; ++i)
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
 * ******************* getPatch ***********************
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
 * ******************* pcaFit ***********************
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
 * ******************* pca_transform ***********************
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
 * ******************* GetFeaturesMaps ***********************
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
 * **************** GetModelOutputsExample ****************
 */
inline std::vector<torch::Tensor>
GetModelOutputsExample(std::vector<itk::ImpactModelConfiguration> & modelsConfig,
                       const std::string &                          modelType,
                       torch::Device                                device)
{

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
 * ******************* GetPatchIndex ***********************
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
 * ******************* GenerateOutputs ***********************
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
 * ******************* GenerateOutputsAndJacobian ***********************
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
