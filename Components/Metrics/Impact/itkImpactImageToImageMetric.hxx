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

#ifndef _itkImpactImageToImageMetric_hxx
#define _itkImpactImageToImageMetric_hxx

#include "itkImpactImageToImageMetric.h"
#include <vector>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkScaleTransform.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */
template <typename TFixedImage, typename TMovingImage>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ImpactImageToImageMetric()
{
  this->Superclass::SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);
}

/**
 * ********************* UpdateFeaturesMaps ****************************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::UpdateFeaturesMaps()
{
  this->m_fixedFeaturesMaps.clear();
  this->m_movingFeaturesMaps.clear();
  this->m_principal_components.clear();

  auto movingWriter = std::function<void(typename TMovingImage::ConstPointer, torch::Tensor &, const std::string &)>(
    [this](typename TMovingImage::ConstPointer image, torch::Tensor & data, const std::string & filename) {
      unsigned int level = this->GetCurrentLevel();
      using WriterType = itk::ImageFileWriter<FeaturesImageType>;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(this->GetFeatureMapsPath() + "/Moving_" + std::to_string(level) + "_" + filename + ".mha");
      writer->SetInput(ImpactTensorUtils::TensorToImage<TMovingImage, FeaturesImageType>(image, data.unsqueeze(0)));
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing image: " << writer->GetFileName() << " ITK Exception: " << error);
      }
    });

  auto fixedWriter = std::function<void(typename TFixedImage::ConstPointer, torch::Tensor &, const std::string &)>(
    [this](typename TFixedImage::ConstPointer image, torch::Tensor & data, const std::string & filename) {
      unsigned int level = this->GetCurrentLevel();
      using WriterType = itk::ImageFileWriter<FeaturesImageType>;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(this->GetFeatureMapsPath() + "/Fixed_" + std::to_string(level) + "_" + filename + ".mha");
      writer->SetInput(ImpactTensorUtils::TensorToImage<TFixedImage, FeaturesImageType>(image, data.unsqueeze(0)));
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing image: " << writer->GetFileName() << " ITK Exception: " << error);
      }
    });

  this->m_movingFeaturesMaps =
    ImpactTensorUtils::GetFeaturesMaps<TMovingImage, FeaturesMaps, InterpolatorType, FeaturesImageType>(
      Superclass::m_MovingImage,
      Superclass::m_Interpolator,
      this->GetMovingModelsConfiguration(),
      this->GetDevice(),
      this->GetPCA(),
      this->m_principal_components,
      this->GetWriteFeatureMaps() ? movingWriter : nullptr);

  this->m_fixedFeaturesMaps =
    ImpactTensorUtils::GetFeaturesMaps<TFixedImage, FeaturesMaps, InterpolatorType, FeaturesImageType>(
      Superclass::m_FixedImage,
      this->m_fixedInterpolator,
      this->GetFixedModelsConfiguration(),
      this->GetDevice(),
      this->GetPCA(),
      this->m_principal_components,
      this->GetWriteFeatureMaps() ? fixedWriter : nullptr);

  if (this->GetWriteFeatureMaps())
  {
    unsigned int level = this->GetCurrentLevel();
    using WriterType = itk::ImageFileWriter<FeaturesImageType>;
    for (int i = 0; i < m_movingFeaturesMaps.size(); ++i)
    {
      typename WriterType::Pointer writer = WriterType::New();

      writer->SetFileName(this->GetFeatureMapsPath() + "/Moving_" + std::to_string(level) + "_" + std::to_string(i) +
                          ".mha");
      writer->SetInput(this->m_movingFeaturesMaps[i].m_featuresMaps);
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing image: " << writer->GetFileName() << " ITK Exception: " << error);
      }
    }
    for (int i = 0; i < m_fixedFeaturesMaps.size(); ++i)
    {
      typename WriterType::Pointer writer = WriterType::New();

      writer->SetFileName(this->GetFeatureMapsPath() + "/Fixed_" + std::to_string(level) + "_" + std::to_string(i) +
                          ".mha");
      writer->SetInput(this->m_fixedFeaturesMaps[i].m_featuresMaps);
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing feature map: " << writer->GetFileName() << " ITK Exception: " << error);
      }
    }
  }
} // end UpdateFeaturesMaps

/**
 * ********************* UpdateMovingFeaturesMaps ****************************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::UpdateMovingFeaturesMaps()
{
  auto movingWriter = std::function<void(typename TMovingImage::ConstPointer, torch::Tensor &, const std::string &)>(
    [this](typename TMovingImage::ConstPointer image, torch::Tensor & data, const std::string & filename) {
      unsigned int level = this->GetCurrentLevel();
      using WriterType = itk::ImageFileWriter<FeaturesImageType>;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(this->GetFeatureMapsPath() + "/Moving_" + std::to_string(level) + "_" + filename + ".mha");
      writer->SetInput(ImpactTensorUtils::TensorToImage<TMovingImage, FeaturesImageType>(image, data.unsqueeze(0)));
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing image: " << writer->GetFileName() << "ITK Exception: " << error);
      }
    });

  this->m_movingFeaturesMaps.clear();
  this->m_movingFeaturesMaps =
    ImpactTensorUtils::GetFeaturesMaps<TMovingImage, FeaturesMaps, InterpolatorType, FeaturesImageType>(
      Superclass::m_MovingImage,
      Superclass::m_Interpolator,
      this->GetMovingModelsConfiguration(),
      this->GetDevice(),
      this->GetPCA(),
      this->m_principal_components,
      this->GetWriteFeatureMaps() ? movingWriter : nullptr,
      std::function<typename TMovingImage::PointType(const typename TMovingImage::PointType &)>(
        [this](const typename TMovingImage::PointType & point) { return this->TransformPoint(point); }));

  using WriterType = itk::ImageFileWriter<FeaturesImageType>;
  if (this->GetWriteFeatureMaps())
  {
    unsigned int level = this->GetCurrentLevel();

    for (int i = 0; i < m_movingFeaturesMaps.size(); ++i)
    {
      typename WriterType::Pointer writer = WriterType::New();

      writer->SetFileName(this->GetFeatureMapsPath() + "/Moving_" + std::to_string(level) + "_" + std::to_string(i) +
                          ".mha");
      writer->SetInput(this->m_movingFeaturesMaps[i].m_featuresMaps);
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing feature map: " << writer->GetFileName() << "ITK Exception: " << error);
      }
    }
  }
} // end UpdateMovingFeaturesMaps

/**
 * ********************* Initialize ****************************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();
  this->m_fixedInterpolator->SetInputImage(Superclass::m_FixedImage);
  this->m_features_indexes.clear();

  if (this->GetMode() == "Static")
  {
    this->UpdateFeaturesMaps();

    if (this->m_fixedFeaturesMaps.size() != this->m_movingFeaturesMaps.size())
    {
      itkExceptionMacro("Mismatch in number of feature maps: "
                        << "fixedFeaturesMaps.size() = " << this->m_fixedFeaturesMaps.size()
                        << ", movingFeaturesMaps.size() = " << this->m_movingFeaturesMaps.size());
    }

    for (int i = 0; i < this->m_fixedFeaturesMaps.size(); ++i)
    {
      if (this->m_fixedFeaturesMaps[i].m_featuresMaps->GetNumberOfComponentsPerPixel() !=
          this->m_movingFeaturesMaps[i].m_featuresMaps->GetNumberOfComponentsPerPixel())
      {
        itkExceptionMacro(
          "Mismatch in number of components per feature map at layer "
          << i << ": fixed = " << this->m_fixedFeaturesMaps[i].m_featuresMaps->GetNumberOfComponentsPerPixel()
          << ", moving = " << this->m_movingFeaturesMaps[i].m_featuresMaps->GetNumberOfComponentsPerPixel());
      }
    }

    for (int i = 0; i < this->m_fixedFeaturesMaps.size(); ++i)
    {
      int numComponents = this->m_fixedFeaturesMaps[i].m_featuresMaps->GetNumberOfComponentsPerPixel();
      this->m_SubsetFeatures[i] = std::clamp<unsigned int>(this->GetSubsetFeatures()[i], 1, numComponents);
      this->m_features_indexes.push_back(std::vector<unsigned int>(numComponents));
      std::iota(this->m_features_indexes[i].begin(), this->m_features_indexes[i].end(), 0);
    }
  }
  else
  {
    std::vector<torch::Tensor> fixedOutputsTensor =
      ImpactTensorUtils::GetModelOutputsExample(this->m_FixedModelsConfiguration, "fixed", this->GetDevice());
    std::vector<torch::Tensor> movingOutputsTensor =
      ImpactTensorUtils::GetModelOutputsExample(this->m_MovingModelsConfiguration, "moving", this->GetDevice());

    if (fixedOutputsTensor.size() != movingOutputsTensor.size())
    {
      itkExceptionMacro("Mismatch in number of feature maps: " << "fixed = " << fixedOutputsTensor.size()
                                                               << ", moving = " << movingOutputsTensor.size());
    }

    for (int i = 0; i < fixedOutputsTensor.size(); ++i)
    {
      if (fixedOutputsTensor[i].size(0) != movingOutputsTensor[i].size(0))
      {
        itkExceptionMacro("Mismatch in number of components per feature map at layer "
                          << i << ": fixed = " << fixedOutputsTensor[i].size(0)
                          << ", moving = " << movingOutputsTensor[i].size(0));
      }
    }

    for (int i = 0; i < fixedOutputsTensor.size(); ++i)
    {
      int numComponents = fixedOutputsTensor[i].size(1);
      this->m_SubsetFeatures[i] = std::clamp<unsigned int>(this->m_SubsetFeatures[i], 1, numComponents);
      this->m_features_indexes.push_back(std::vector<unsigned int>(numComponents));
      std::iota(this->m_features_indexes[i].begin(), this->m_features_indexes[i].end(), 0);
    }
  }
} // end Initialize

/**
 * ******************* SampleCheck *******************
 */
template <typename TFixedImage, typename TMovingImage>
bool
ImpactImageToImageMetric<TFixedImage, TMovingImage>::SampleCheck(
  const FixedImagePointType & fixedImageCenterCoordinate) const
{
  FixedImagePointType  fixedImagePoint(fixedImageCenterCoordinate);
  MovingImagePointType mappedPoint;
  mappedPoint = this->TransformPoint(fixedImagePoint);
  if (Superclass::m_Interpolator->IsInsideBuffer(mappedPoint) == false)
  {
    return false;
  }
  else
  {
    if (const auto * const mask = this->GetMovingImageMask())
    {
      if (mask->IsInsideInWorldSpace(mappedPoint) == false)
      {
        return false;
      }
    }
  }
  return true;
} // end SampleCheck

/**
 * ******************* SampleCheck *******************
 */
template <typename TFixedImage, typename TMovingImage>
bool
ImpactImageToImageMetric<TFixedImage, TMovingImage>::SampleCheck(
  const FixedImagePointType &             fixedImageCenterCoordinate,
  const std::vector<std::vector<float>> & patchIndex) const
{
  FixedImagePointType  fixedImagePoint(fixedImageCenterCoordinate);
  MovingImagePointType mappedPoint;
  for (const std::vector<float> & patchIndexItem : patchIndex)
  {
    for (int dim = 0; dim < patchIndexItem.size(); ++dim)
    {
      fixedImagePoint[dim] = fixedImageCenterCoordinate[dim] + patchIndexItem[dim];
    }
    mappedPoint = this->TransformPoint(fixedImagePoint);
    if (Superclass::m_Interpolator->IsInsideBuffer(mappedPoint) == false)
    {
      return false;
    }
    else
    {
      if (const auto * const mask = this->GetMovingImageMask())
      {
        if (mask->IsInsideInWorldSpace(mappedPoint) == false)
        {
          return false;
        }
      }
    }
  }
  return true;
} // end SampleCheck

/**
 * ******************* GeneratePatchIndex *******************
 */
template <typename TFixedImage, typename TMovingImage>
template <typename ImagePointType>
std::vector<ImagePointType>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GeneratePatchIndex(
  const std::vector<ImpactModelConfiguration> &               modelConfig,
  std::mt19937 &                                              randomGenerator,
  const std::vector<ImagePointType> &                         fixedPointsTmp,
  std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex) const
{

  std::vector<ImagePointType> fixedPoints;

  std::vector<bool> pointsMask(fixedPointsTmp.size(), true);
  for (int i = 0; i < modelConfig.size(); ++i)
  {
    patchIndex[i] = std::vector<std::vector<std::vector<float>>>();
    for (int it = 0; it < fixedPointsTmp.size(); ++it)
    {
      std::vector<std::vector<float>> patch =
        ImpactTensorUtils::GetPatchIndex(modelConfig[i], randomGenerator, FixedImageDimension);
      if (this->SampleCheck(fixedPointsTmp[it], patch))
      {
        patchIndex[i].push_back(patch);
      }
      else
      {
        pointsMask[it] = false;
      }
    }
  }
  for (int it = 0; it < fixedPointsTmp.size(); ++it)
  {
    if (pointsMask[it])
    {
      fixedPoints.push_back(fixedPointsTmp[it]);
    }
  }
  return fixedPoints;
} // end GeneratePatchIndex

/**
 * ******************* EvaluateFixedImagesPatchValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
torch::Tensor
ImpactImageToImageMetric<TFixedImage, TMovingImage>::EvaluateFixedImagesPatchValue(
  const FixedImagePointType &             fixedImageCenterCoordinate,
  const std::vector<std::vector<float>> & patchIndex,
  const std::vector<int64_t> &            patchSize) const
{
  std::vector<float> fixedImagesPatchValues(patchIndex.size(), 0.0f);

  FixedImagePointType fixedImagePoint(fixedImageCenterCoordinate);
  for (int i = 0; i < patchIndex.size(); ++i)
  {
    for (int dim = 0; dim < patchIndex[i].size(); ++dim)
    {
      fixedImagePoint[dim] = fixedImageCenterCoordinate[dim] + patchIndex[i][dim];
    }
    fixedImagesPatchValues[i] = this->m_fixedInterpolator->Evaluate(fixedImagePoint);
  }
  return torch::from_blob(fixedImagesPatchValues.data(), { torch::IntArrayRef(patchSize) }, torch::kFloat32)
    .unsqueeze(0)
    .clone();
} // end EvaluateFixedImagesPatchValue

/**
 * ******************* EvaluateFixedPatchValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
torch::Tensor
ImpactImageToImageMetric<TFixedImage, TMovingImage>::EvaluateMovingImagesPatchValue(
  const FixedImagePointType &             fixedImageCenterCoordinate,
  const std::vector<std::vector<float>> & patchIndex,
  const std::vector<int64_t> &            patchSize) const
{
  std::vector<float> movingImagesPatchValues(patchIndex.size(), 0.0f);
  RealType           movingImageValue;

  FixedImagePointType fixedImagePoint(fixedImageCenterCoordinate);
  for (int i = 0; i < patchIndex.size(); ++i)
  {
    for (int dim = 0; dim < patchIndex[i].size(); ++dim)
    {
      fixedImagePoint[dim] = fixedImageCenterCoordinate[dim] + patchIndex[i][dim];
    }
    this->Superclass::EvaluateMovingImageValueAndDerivative(
      this->TransformPoint(fixedImagePoint), movingImageValue, nullptr);
    movingImagesPatchValues[i] = movingImageValue;
  }
  return torch::from_blob(movingImagesPatchValues.data(), { torch::IntArrayRef(patchSize) }, torch::kFloat32)
    .clone()
    .unsqueeze(0);
} // end EvaluateFixedPatchValue

template <typename TFixedImage, typename TMovingImage>
std::vector<unsigned int>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetSubsetOfFeatures(
  const std::vector<unsigned int> & features_index,
  std::mt19937 &                    randomGenerator,
  int                               n) const
{
  if (features_index.size() == static_cast<size_t>(n))
    return features_index;

  std::vector<unsigned int> shuffled = features_index;
  std::shuffle(shuffled.begin(), shuffled.end(), randomGenerator);
  shuffled.resize(n);
  return shuffled;
}

/**
 * ******************* EvaluateMovingPatchValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
torch::Tensor
ImpactImageToImageMetric<TFixedImage, TMovingImage>::EvaluateMovingImagesPatchValuesAndJacobians(
  const FixedImagePointType &             fixedImageCenterCoordinate,
  torch::Tensor &                         movingImagesPatchesJacobians,
  const std::vector<std::vector<float>> & patchIndex,
  const std::vector<int64_t> &            patchSize,
  int                                     s) const
{

  std::vector<float> movingImagesPatchValues(patchIndex.size(), 0.0f);
  std::vector<float> movingImagesPatchJacobians(patchIndex.size() * MovingImageDimension, 0.0f);

  RealType                  movingImageValue;
  MovingImageDerivativeType movingImageJacobian;

  FixedImagePointType fixedImagePoint(fixedImageCenterCoordinate);
  for (int i = 0; i < patchIndex.size(); ++i)
  {
    for (int dim = 0; dim < patchIndex[i].size(); ++dim)
    {
      fixedImagePoint[dim] = fixedImageCenterCoordinate[dim] + patchIndex[i][dim];
    }
    this->Superclass::EvaluateMovingImageValueAndDerivative(
      this->TransformPoint(fixedImagePoint), movingImageValue, &movingImageJacobian);
    movingImagesPatchValues[i] = movingImageValue;
    for (unsigned int it = 0; it < MovingImageDimension; ++it)
    {
      movingImagesPatchJacobians[i * MovingImageDimension + it] = static_cast<float>(movingImageJacobian[it]);
    }
  }
  movingImagesPatchesJacobians[s] = torch::from_blob(movingImagesPatchJacobians.data(),
                                                     { static_cast<int64_t>(patchIndex.size()), MovingImageDimension },
                                                     torch::kFloat32)
                                      .clone();
  return torch::from_blob(movingImagesPatchValues.data(), { torch::IntArrayRef(patchSize) }, torch::kFloat32)
    .unsqueeze(0)
    .clone();
} // end EvaluateMovingPatchValueAndDerivative


/**
 * ******************* ComputeValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
unsigned int
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValue(
  const std::vector<FixedImagePointType> & fixedPointsTmp,
  LossPerThreadStruct &                    loss) const
{
  std::vector<std::vector<std::vector<std::vector<float>>>> patchIndex(this->GetFixedModelsConfiguration().size());
  std::vector<FixedImagePointType>                          fixedPoints = this->GeneratePatchIndex<FixedImagePointType>(
    this->GetFixedModelsConfiguration(), loss.m_randomGenerator, fixedPointsTmp, patchIndex);
  if (fixedPoints.empty())
  {
    return 0;
  }
  unsigned int               nb_sample = fixedPoints.size();
  std::vector<torch::Tensor> fixedOutputsTensor, movingOutputsTensor;

  std::vector<torch::Tensor> subsetsOfFeatures(this->m_features_indexes.size());

  for (int i = 0; i < this->m_features_indexes.size(); ++i)
  {
    std::vector<unsigned int> subsetOfFeatures =
      this->GetSubsetOfFeatures(this->m_features_indexes[i], loss.m_randomGenerator, this->GetSubsetFeatures()[i]);
    subsetsOfFeatures[i] =
      torch::from_blob(subsetOfFeatures.data(), { static_cast<long>(subsetOfFeatures.size()) }, torch::kUInt32)
        .to(torch::kUInt64)
        .to(this->GetDevice())
        .clone();
  }

  const ImpactTensorUtils::ImagesPatchValuesEvaluator<FixedImagePointType> fixedimagesPatchValuesEvaluator =
    [this](const FixedImagePointType &             fixedImageCenterCoordinateLoc,
           const std::vector<std::vector<float>> & patchIndexLoc,
           const std::vector<int64_t> &            patchSizeLoc) {
      return this->EvaluateFixedImagesPatchValue(fixedImageCenterCoordinateLoc, patchIndexLoc, patchSizeLoc);
    };


  fixedOutputsTensor = ImpactTensorUtils::GenerateOutputs<FixedImagePointType>(this->GetFixedModelsConfiguration(),
                                                                               fixedPoints,
                                                                               patchIndex,
                                                                               subsetsOfFeatures,
                                                                               this->GetDevice(),
                                                                               fixedimagesPatchValuesEvaluator);

  const ImpactTensorUtils::ImagesPatchValuesEvaluator<FixedImagePointType> movingimagesPatchValuesEvaluator =
    [this](const MovingImagePointType &            fixedImageCenterCoordinateLoc,
           const std::vector<std::vector<float>> & patchIndexLoc,
           const std::vector<int64_t> &            patchSizeLoc) {
      return this->EvaluateMovingImagesPatchValue(fixedImageCenterCoordinateLoc, patchIndexLoc, patchSizeLoc);
    };

  movingOutputsTensor = ImpactTensorUtils::GenerateOutputs<MovingImagePointType>(this->GetMovingModelsConfiguration(),
                                                                                 fixedPoints,
                                                                                 patchIndex,
                                                                                 subsetsOfFeatures,
                                                                                 this->GetDevice(),
                                                                                 movingimagesPatchValuesEvaluator);

  for (int i = 0; i < fixedOutputsTensor.size(); ++i)
  {
    loss.m_losses[i]->updateValue(fixedOutputsTensor[i], movingOutputsTensor[i]);
  }
  return nb_sample;
} // end ComputeValue

/**
 * ******************* ComputeValueStatic *******************
 */
template <typename TFixedImage, typename TMovingImage>
unsigned int
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueStatic(
  const std::vector<FixedImagePointType> & fixedPointsTmp,
  LossPerThreadStruct &                    loss) const
{
  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(fixedPointsTmp.size());
  for (FixedImagePointType fixedPoint : fixedPointsTmp)
  {
    if (this->SampleCheck(fixedPoint))
    {
      fixedPoints.push_back(fixedPoint);
    }
  }
  if (fixedPoints.empty())
  {
    return 0;
  }
  unsigned int nb_sample = fixedPoints.size();
  for (int i = 0; i < this->m_fixedFeaturesMaps.size(); ++i)
  {
    std::vector<unsigned int> subsetOfFeatures =
      this->GetSubsetOfFeatures(this->m_features_indexes[i], loss.m_randomGenerator, this->GetSubsetFeatures()[i]);

    torch::Tensor fixedOutputTensor = torch::zeros({ nb_sample, this->GetSubsetFeatures()[i] });
    torch::Tensor movingOutputTensor = torch::zeros({ nb_sample, this->GetSubsetFeatures()[i] });
    for (unsigned int s = 0; s < nb_sample; ++s)
    {
      const auto & fixedPoint = fixedPoints[s];
      fixedOutputTensor[s] =
        this->m_fixedFeaturesMaps[i].m_featuresMapsInterpolator.Evaluate(fixedPoint, subsetOfFeatures);
      movingOutputTensor[s] = this->m_movingFeaturesMaps[i].m_featuresMapsInterpolator.Evaluate(
        this->TransformPoint(fixedPoint), subsetOfFeatures);
    }
    loss.m_losses[i]->updateValue(fixedOutputTensor, movingOutputTensor);
  }
  return nb_sample;
} // end ComputeValueStatic

/**
 * ******************* ComputeValueAndDerivativeJacobian *******************
 */
template <typename TFixedImage, typename TMovingImage>
unsigned int
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeJacobian(
  const std::vector<FixedImagePointType> & fixedPointsTmp,
  LossPerThreadStruct &                    loss) const
{

  std::vector<std::vector<std::vector<std::vector<float>>>> patchIndex(this->GetFixedModelsConfiguration().size());
  std::vector<FixedImagePointType>                          fixedPoints = this->GeneratePatchIndex<FixedImagePointType>(
    this->GetFixedModelsConfiguration(), loss.m_randomGenerator, fixedPointsTmp, patchIndex);
  if (fixedPoints.empty())
  {
    return 0;
  }
  unsigned int  nb_sample = fixedPoints.size();
  const int     numNonZeroJacobianIndices = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  torch::Tensor nonZeroJacobianIndices = torch::zeros({ nb_sample, numNonZeroJacobianIndices }, torch::kLong);
  torch::Tensor transformsJacobian =
    torch::zeros({ nb_sample, MovingImageDimension, static_cast<int64_t>(numNonZeroJacobianIndices) }, torch::kFloat32);

  TransformJacobianType      flatTransformJacobian; // class itk::Array2D<double>
  NonZeroJacobianIndicesType flatNonZeroJacobianIndices(
    numNonZeroJacobianIndices); // class std::vector<unsigned long,class std::allocator<uint64_t> >

  for (unsigned int s = 0; s < nb_sample; ++s)
  {
    this->m_AdvancedTransform->GetJacobian(fixedPoints[s], flatTransformJacobian, flatNonZeroJacobianIndices);
    nonZeroJacobianIndices[s] =
      torch::from_blob(&(*flatNonZeroJacobianIndices.begin()), { numNonZeroJacobianIndices }, torch::kUInt64)
        .to(torch::kLong)
        .clone();

    transformsJacobian[s] = torch::from_blob(&(*flatTransformJacobian.begin()),
                                             { MovingImageDimension, numNonZeroJacobianIndices },
                                             torch::kDouble)
                              .to(torch::kFloat32)
                              .clone();
  }
  transformsJacobian = transformsJacobian.to(this->GetDevice());
  nonZeroJacobianIndices = nonZeroJacobianIndices.to(this->GetDevice());
  std::vector<torch::Tensor> subsetsOfFeatures(this->m_features_indexes.size());

  for (int i = 0; i < this->m_features_indexes.size(); ++i)
  {
    std::vector<unsigned int> subsetOfFeatures =
      this->GetSubsetOfFeatures(this->m_features_indexes[i], loss.m_randomGenerator, this->GetSubsetFeatures()[i]);
    subsetsOfFeatures[i] =
      torch::from_blob(subsetOfFeatures.data(), { static_cast<long>(subsetOfFeatures.size()) }, torch::kUInt32)
        .to(torch::kInt64)
        .to(this->GetDevice())
        .clone();
  }

  const ImpactTensorUtils::ImagesPatchValuesEvaluator<FixedImagePointType> imagesPatchValuesEvaluator =
    [this](const FixedImagePointType &             fixedImageCenterCoordinateLoc,
           const std::vector<std::vector<float>> & patchIndexLoc,
           const std::vector<int64_t> &            patchSizeLoc) {
      return this->EvaluateFixedImagesPatchValue(fixedImageCenterCoordinateLoc, patchIndexLoc, patchSizeLoc);
    };

  std::vector<torch::Tensor> fixedOutputsTensor, movingOutputsTensor;
  fixedOutputsTensor = ImpactTensorUtils::GenerateOutputs<FixedImagePointType>(this->GetFixedModelsConfiguration(),
                                                                               fixedPoints,
                                                                               patchIndex,
                                                                               subsetsOfFeatures,
                                                                               this->GetDevice(),
                                                                               imagesPatchValuesEvaluator);


  const ImpactTensorUtils::ImagesPatchValuesAndJacobiansEvaluator<MovingImagePointType>
    imagesPatchValuesAndJacobiansEvaluator = [this](const MovingImagePointType & fixedImageCenterCoordinateLoc,
                                                    torch::Tensor &              movingImagesPatchesJacobiansLoc,
                                                    const std::vector<std::vector<float>> & patchIndexLoc,
                                                    const std::vector<int64_t> &            patchSizeLoc,
                                                    int                                     sLoc) {
      return this->EvaluateMovingImagesPatchValuesAndJacobians(
        fixedImageCenterCoordinateLoc, movingImagesPatchesJacobiansLoc, patchIndexLoc, patchSizeLoc, sLoc);
    };

  std::vector<torch::Tensor> layersJacobian =
    ImpactTensorUtils::GenerateOutputsAndJacobian<MovingImagePointType>(this->GetMovingModelsConfiguration(),
                                                                        fixedPoints,
                                                                        patchIndex,
                                                                        subsetsOfFeatures,
                                                                        fixedOutputsTensor,
                                                                        this->GetDevice(),
                                                                        loss.m_losses,
                                                                        imagesPatchValuesAndJacobiansEvaluator);

  for (int i = 0; i < fixedOutputsTensor.size(); ++i)
  {
    torch::Tensor jacobian = torch::bmm(layersJacobian[i], transformsJacobian);
    loss.m_losses[i]->updateDerivativeInJacobianMode(jacobian, nonZeroJacobianIndices);
  }
  return nb_sample;
} // end ComputeValueAndDerivativeJacobian

/**
 * ******************* ComputeValueAndDerivativeStatic *******************
 */
template <typename TFixedImage, typename TMovingImage>
unsigned int
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeStatic(
  const std::vector<FixedImagePointType> & fixedPointsTmp,
  LossPerThreadStruct &                    loss) const
{
  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(fixedPointsTmp.size());
  for (FixedImagePointType fixedPoint : fixedPointsTmp)
  {
    if (this->SampleCheck(fixedPoint))
    {
      fixedPoints.push_back(fixedPoint);
    }
  }
  if (fixedPoints.empty())
  {
    return 0;
  }
  unsigned int  nb_sample = fixedPoints.size();
  const int     numNonZeroJacobianIndices = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  torch::Tensor nonZeroJacobianIndices = torch::zeros({ nb_sample, numNonZeroJacobianIndices }, torch::kLong);

  torch::Tensor transformsJacobian =
    torch::zeros({ nb_sample, MovingImageDimension, static_cast<int64_t>(numNonZeroJacobianIndices) }, torch::kFloat32);

  TransformJacobianType      flatTransformJacobian; // class itk::Array2D<double>
  NonZeroJacobianIndicesType flatNonZeroJacobianIndices(
    numNonZeroJacobianIndices); // class std::vector<unsigned long,class std::allocator<uint64_t> >
  for (unsigned int s = 0; s < nb_sample; ++s)
  {
    this->m_AdvancedTransform->GetJacobian(fixedPoints[s], flatTransformJacobian, flatNonZeroJacobianIndices);
    nonZeroJacobianIndices[s] =
      torch::from_blob(&(*flatNonZeroJacobianIndices.begin()), { numNonZeroJacobianIndices }, torch::kUInt64)
        .to(torch::kLong)
        .clone();
    transformsJacobian[s] = torch::from_blob(&(*flatTransformJacobian.begin()),
                                             { MovingImageDimension, numNonZeroJacobianIndices },
                                             torch::kDouble)
                              .to(torch::kFloat32)
                              .clone();
  }


  for (int i = 0; i < this->m_fixedFeaturesMaps.size(); ++i)
  {
    std::vector<unsigned int> subsetOfFeatures =
      this->GetSubsetOfFeatures(this->m_features_indexes[i], loss.m_randomGenerator, this->GetSubsetFeatures()[i]);

    MovingImagePointType mappedPoint;
    torch::Tensor        fixedOutputTensor = torch::zeros({ nb_sample, this->GetSubsetFeatures()[i] }, torch::kFloat32);
    torch::Tensor movingOutputTensor = torch::zeros({ nb_sample, this->GetSubsetFeatures()[i] }, torch::kFloat32);
    torch::Tensor movingDerivativeTensor =
      torch::zeros({ nb_sample, this->GetSubsetFeatures()[i], MovingImageDimension }, torch::kFloat32);
    for (unsigned int s = 0; s < nb_sample; ++s)
    {
      const auto & fixedPoint = fixedPoints[s];
      mappedPoint = this->TransformPoint(fixedPoint);
      fixedOutputTensor[s] =
        this->m_fixedFeaturesMaps[i].m_featuresMapsInterpolator.Evaluate(fixedPoint, subsetOfFeatures);
      movingOutputTensor[s] =
        this->m_movingFeaturesMaps[i].m_featuresMapsInterpolator.Evaluate(mappedPoint, subsetOfFeatures);
      movingDerivativeTensor[s] =
        this->m_movingFeaturesMaps[i].m_featuresMapsInterpolator.EvaluateDerivative(mappedPoint, subsetOfFeatures);
    }
    torch::Tensor jacobian = torch::bmm(movingDerivativeTensor, transformsJacobian);
    loss.m_losses[i]->updateValueAndDerivativeInStaticMode(
      fixedOutputTensor, movingOutputTensor, jacobian, nonZeroJacobianIndices);
  }
  return nb_sample;
} // end ComputeValueAndDerivativeStatic

/**
 * ******************* InitializeThreadingParameters *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  this->m_LossThreadStruct.reset(new AlignedLossPerThreadStruct[numberOfThreads]);
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    this->m_LossThreadStruct[i].init(this->GetDistance(), this->GetLayersWeight(), this->m_Seed);
  }
  this->m_LossThreadStructSize = numberOfThreads;

  const int nb_parameters = this->GetNumberOfParameters();
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    this->m_LossThreadStruct[i].set_nb_parameters(nb_parameters);
  }

} // end InitializeThreadingParameters

/**
 * ******************* GetValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
auto
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValue(const ParametersType & parameters) const -> MeasureType
{
  if (!this->m_UseMultiThread)
  {
    return this->GetValueSingleThreaded(parameters);
  }
  this->BeforeThreadedGetValueAndDerivative(parameters);
  this->LaunchGetValueThreaderCallback();
  MeasureType value{};
  this->AfterThreadedGetValue(value);
  return value;
} // end GetValue

/**
 * ******************* GetValueSingleThreaded *******************
 */
template <typename TFixedImage, typename TMovingImage>
auto
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueSingleThreaded(const ParametersType & parameters) const
  -> MeasureType
{
  this->BeforeThreadedGetValueAndDerivative(parameters);
  /** Initialize some variables. */
  auto & loss = this->m_LossThreadStruct[0];
  loss.reset();

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->size();

  auto computeValueFunc = (this->GetMode() == "Static")
                            ? &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueStatic
                            : &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValue;

  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(sampleContainerSize);
  for (unsigned int i = 0; i < sampleContainerSize; ++i)
  {
    fixedPoints.push_back((*sampleContainer)[i].m_ImageCoordinates);
  }
  Superclass::m_NumberOfPixelsCounted = (this->*computeValueFunc)(fixedPoints, loss);
  this->CheckNumberOfSamples();

  return loss.GetValue();
} // end GetValueSingleThreaded

/**
 * ******************* ThreadedGetValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValue(ThreadIdType threadId) const
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto pos_begin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto pos_end = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto threader_fbegin = beginOfSampleContainer + pos_begin;
  const auto threader_fend = beginOfSampleContainer + pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */

  LossPerThreadStruct & loss = this->m_LossThreadStruct[threadId];
  loss.reset();
  auto                             computeValueFunc = (this->GetMode() == "Static")
                                                        ? &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueStatic
                                                        : &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValue;
  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(nrOfSamplesPerThreads);
  for (auto threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    fixedPoints.push_back(threader_fiter->m_ImageCoordinates);
  }
  loss.m_numberOfPixelsCounted += (this->*computeValueFunc)(fixedPoints, loss);
} // end ThreadedGetValue

/**
 * ******************* AfterThreadedGetValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValue(MeasureType & value) const
{
  LossPerThreadStruct & loss = this->m_LossThreadStruct[0];
  for (ThreadIdType i = 1; i < Self::GetNumberOfWorkUnits(); ++i)
  {
    loss += this->m_LossThreadStruct[i];
  }
  Superclass::m_NumberOfPixelsCounted = loss.m_numberOfPixelsCounted;
  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  /** Accumulate values. */
  value = loss.GetValue();
} // end AfterThreadedGetValue

/**
 * ******************* GetDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(const ParametersType & parameters,
                                                                   DerivativeType &       derivative) const
{
  MeasureType dummyvalue{};
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);
} // end GetDerivative


/**
 * ******************* GetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(const ParametersType & parameters,
                                                                           MeasureType &          value,
                                                                           DerivativeType &       derivative) const
{

  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->GetValueAndDerivativeSingleThreaded(parameters, value, derivative);
  }

  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Launch multi-threading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative(value, derivative);

} // end GetValueAndDerivative

/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{

  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Initialize some variables. */
  LossPerThreadStruct & loss = this->m_LossThreadStruct[0];
  loss.reset();

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->size();

  auto computeValueAndDerivativeFunc = [this](const std::string & mode) {
    if (mode == "Jacobian")
    {
      return &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeJacobian;
    }
    else
    {
      return &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeStatic;
    }
  }(this->GetMode());

  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(sampleContainerSize);
  for (unsigned int i = 0; i < sampleContainerSize; ++i)
  {
    fixedPoints.push_back((*sampleContainer)[i].m_ImageCoordinates);
  }
  Superclass::m_NumberOfPixelsCounted = (this->*computeValueAndDerivativeFunc)(fixedPoints, loss);
  this->CheckNumberOfSamples();

  value = loss.GetValue();
  derivative = loss.GetDerivative();
} // end GetValueAndDerivativeSingleThreaded

/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValueAndDerivative(ThreadIdType threadId) const
{
  LossPerThreadStruct & loss = this->m_LossThreadStruct[threadId];
  loss.reset();

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();
  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto pos_begin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto pos_end = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto threader_fbegin = beginOfSampleContainer + pos_begin;
  const auto threader_fend = beginOfSampleContainer + pos_end;
  auto       computeValueAndDerivativeFunc = [this](const std::string & mode) {
    if (mode == "Jacobian")
    {
      return &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeJacobian;
    }
    else
    {
      return &ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeStatic;
    }
  }(this->GetMode());

  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(nrOfSamplesPerThreads);
  for (auto threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    fixedPoints.push_back(threader_fiter->m_ImageCoordinates);
  }
  loss.m_numberOfPixelsCounted += (this->*computeValueAndDerivativeFunc)(fixedPoints, loss);
} // end ThreadedGetValueAndDerivative

/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  LossPerThreadStruct & loss = this->m_LossThreadStruct[0];
  for (ThreadIdType i = 1; i < Self::GetNumberOfWorkUnits(); ++i)
  {
    loss += this->m_LossThreadStruct[i];
  }
  Superclass::m_NumberOfPixelsCounted = loss.m_numberOfPixelsCounted;
  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  /** Accumulate values. */
  value = loss.GetValue();
  derivative = loss.GetDerivative();
} // end AfterThreadedGetValueAndDerivative

} // end namespace itk

#endif // end #ifndef _itkImpactImageToImageMetric_hxx
