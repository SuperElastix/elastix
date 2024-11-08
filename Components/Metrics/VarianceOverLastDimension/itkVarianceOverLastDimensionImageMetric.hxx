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
#ifndef itkVarianceOverLastDimensionImageMetric_hxx
#define itkVarianceOverLastDimensionImageMetric_hxx

#include "itkVarianceOverLastDimensionImageMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <vnl/algo/vnl_matrix_update.h>
#include <numeric>

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <typename TFixedImage, typename TMovingImage>
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::VarianceOverLastDimensionImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

} // end Constructor


/**
 * ******************* Initialize *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = FixedImageDimension - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Check num last samples. */
  if (m_NumSamplesLastDimension > lastDimSize)
  {
    m_NumSamplesLastDimension = lastDimSize;
  }

  /** Compute variance over last dimension for complete image to use as normalization factor. */
  ImageLinearConstIteratorWithIndex<MovingImageType> it(this->GetMovingImage(),
                                                        this->GetMovingImage()->GetLargestPossibleRegion());
  it.SetDirection(lastDim);
  it.GoToBegin();

  float sumvar = 0.0;
  int   num = 0;
  while (!it.IsAtEnd())
  {
    /** Compute sum of values and sum of squared values. */
    float        sum = 0.0;
    float        sumsq = 0.0;
    unsigned int numlast = 0;
    while (!it.IsAtEndOfLine())
    {
      float value = it.Get();
      sum += value;
      sumsq += value * value;
      ++numlast;
      ++it;
    }

    /** Compute expected value (mean) and variance. */
    float expectedValue = sum / static_cast<float>(numlast);
    sumvar += sumsq / static_cast<float>(numlast) - expectedValue * expectedValue;
    ++num;

    it.NextLine();
  }

  /** Compute average variance. */
  if (sumvar == 0)
  {
    m_InitialVariance = 1.0f;
  }
  else
  {
    m_InitialVariance = sumvar / static_cast<float>(num);
  }

} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* SampleRandom *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::SampleRandom(const int          n,
                                                                              const int          m,
                                                                              std::vector<int> & numbers) const
{
  /** Empty list of last dimension positions. */
  numbers.clear();
  numbers.reserve(m_NumAdditionalSamplesFixed + n);

  /** Initialize random number generator. */
  Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator =
    Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();

  /** Sample additional at fixed timepoint. */
  for (unsigned int i = 0; i < m_NumAdditionalSamplesFixed; ++i)
  {
    numbers.push_back(m_ReducedDimensionIndex);
  }

  /** Get n random samples. */
  for (int i = 0; i < n; ++i)
  {
    int randomNum = 0;
    do
    {
      randomNum = static_cast<int>(randomGenerator->GetVariateWithClosedRange(m));
    } while (find(numbers.begin(), numbers.end(), randomNum) != numbers.end());
    numbers.push_back(randomNum);
  }
} // end SampleRandom()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template <typename TFixedImage, typename TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  ImplementationDetails::EvaluateInnerProduct(jacobian, movingImageDerivative, imageJacobian);

} // end EvaluateTransformJacobianInnerProduct()


/**
 * ******************* GetValue *******************
 */

template <typename TFixedImage, typename TMovingImage>
auto
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::GetValue(const ParametersType & parameters) const
  -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Initialize some variables */
  Superclass::m_NumberOfPixelsCounted = 0;
  MeasureType measure{};

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = FixedImageDimension - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);
  const unsigned int numLastDimSamples = m_NumSamplesLastDimension;

  /** Vector containing last dimension positions to use:
   * initialize on all positions when random sampling turned off.
   */
  std::vector<int> lastDimPositions;
  if (!m_SampleLastDimensionRandomly)
  {
    lastDimPositions.reserve(lastDimSize);

    for (unsigned int i = 0; i < lastDimSize; ++i)
    {
      lastDimPositions.push_back(i);
    }
  }

  /** Loop over the fixed image samples to calculate the variance over time for every sample position. */
  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fixedImageSample.m_ImageCoordinates;

    /** Determine random last dimension positions if needed. */
    if (m_SampleLastDimensionRandomly)
    {
      this->SampleRandom(numLastDimSamples, lastDimSize, lastDimPositions);
    }

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    /** Loop over the slowest varying dimension. */
    float              sumValues = 0.0;
    float              sumValuesSquared = 0.0;
    unsigned int       numSamplesOk = 0;
    const unsigned int realNumLastDimPositions = lastDimPositions.size();
    for (unsigned int d = 0; d < realNumLastDimPositions; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = lastDimPositions[d];

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);

      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      /** Compute the moving image value and check if the point is
       * inside the moving image buffer.
       */
      if (sampleOk)
      {
        sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
      }

      if (sampleOk)
      {
        ++numSamplesOk;
        sumValues += movingImageValue;
        sumValuesSquared += movingImageValue * movingImageValue;
      } // end if sampleOk
    }   // end for loop over last dimension

    if (numSamplesOk > 0)
    {
      Superclass::m_NumberOfPixelsCounted++;

      /** Add this variance to the variance sum. */
      const float expectedValue = sumValues / static_cast<float>(numSamplesOk);
      const float expectedSquaredValue = sumValuesSquared / static_cast<float>(numSamplesOk);
      measure += expectedSquaredValue - expectedValue * expectedValue;
    }

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), Superclass::m_NumberOfPixelsCounted);

  /** Compute average over variances. */
  measure /= static_cast<float>(Superclass::m_NumberOfPixelsCounted);
  /** Normalize with initial variance. */
  measure /= m_InitialVariance;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::GetDerivative(const ParametersType & parameters,
                                                                               DerivativeType &       derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue{};
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Initialize some variables */
  Superclass::m_NumberOfPixelsCounted = 0;
  MeasureType measure{};
  derivative.set_size(this->GetNumberOfParameters());
  derivative.Fill(0.0);

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = FixedImageDimension - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Vector containing last dimension positions to use:
   * initialize on all positions when random sampling turned off.
   */
  std::vector<int> lastDimPositions;
  if (!m_SampleLastDimensionRandomly)
  {
    lastDimPositions.reserve(lastDimSize);

    for (unsigned int i = 0; i < lastDimSize; ++i)
    {
      lastDimPositions.push_back(i);
    }
  }

  /** Create variables to store intermediate results in. */
  TransformJacobianType jacobian;
  DerivativeType        imageJacobian(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());

  /** Get real last dim samples. */
  const unsigned int realNumLastDimPositions =
    m_SampleLastDimensionRandomly ? m_NumSamplesLastDimension + m_NumAdditionalSamplesFixed : lastDimSize;

  /** Variable to store and nzjis. */
  std::vector<NonZeroJacobianIndicesType> nzjis(realNumLastDimPositions, NonZeroJacobianIndicesType());

  std::vector<RealType>       MT(realNumLastDimPositions);
  std::vector<DerivativeType> dMTdmu(realNumLastDimPositions);

  /** Loop over the fixed image samples to calculate the variance over time for every sample position. */
  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fixedImageSample.m_ImageCoordinates;

    /** Determine random last dimension positions if needed. */
    if (m_SampleLastDimensionRandomly)
    {
      this->SampleRandom(m_NumSamplesLastDimension, lastDimSize, lastDimPositions);
    }

    /** Initialize MT vector. */
    std::fill(MT.begin(), MT.end(), RealType{});

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    /** Loop over the slowest varying dimension. */
    float        sumValues = 0.0;
    float        sumValuesSquared = 0.0;
    unsigned int numSamplesOk = 0;

    /** First loop over t: compute M(T(x,t)), dM(T(x,t))/dmu, nzji and store. */
    for (unsigned int d = 0; d < realNumLastDimPositions; ++d)
    {
      /** Initialize some variables. */
      RealType                  movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = lastDimPositions[d];
      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);
      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      /** Compute the moving image value and check if the point is
       * inside the moving image buffer. */
      if (sampleOk)
      {
        sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, &movingImageDerivative);
      }

      if (sampleOk)
      {
        /** Update value terms **/
        ++numSamplesOk;
        sumValues += movingImageValue;
        sumValuesSquared += movingImageValue * movingImageValue;

        /** Get the TransformJacobian dT/dmu. */
        this->EvaluateTransformJacobian(fixedPoint, jacobian, nzjis[d]);

        /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
        this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

        /** Store values. */
        MT[d] = movingImageValue;
        dMTdmu[d] = imageJacobian;
      }
      else
      {
        dMTdmu[d] = DerivativeType(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
        dMTdmu[d].Fill(0.0);
        nzjis[d] = NonZeroJacobianIndicesType(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices(), 0);
      } // end if sampleOk
    }

    if (numSamplesOk > 0)
    {
      Superclass::m_NumberOfPixelsCounted++;

      /** Compute average intensity value. */
      const float expectedValue = sumValues / static_cast<float>(numSamplesOk);
      /** Add this variance to the variance sum. */
      const float expectedSquaredValue = sumValuesSquared / static_cast<float>(numSamplesOk);
      measure += expectedSquaredValue - expectedValue * expectedValue;

      /** Second loop over t: update derivative. */
      for (unsigned int d = 0; d < realNumLastDimPositions; ++d)
      {
        for (unsigned int j = 0; j < nzjis[d].size(); ++j)
        {
          derivative[nzjis[d][j]] += (2.0 * (MT[d] - expectedValue) * dMTdmu[d][j]) / static_cast<float>(numSamplesOk);
        }
      }
    }
  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), Superclass::m_NumberOfPixelsCounted);

  /** Compute average over variances and normalize with initial variance. */
  measure /= static_cast<float>(Superclass::m_NumberOfPixelsCounted * m_InitialVariance);
  derivative /= static_cast<float>(Superclass::m_NumberOfPixelsCounted * m_InitialVariance);

  /** Subtract mean from derivative elements. */
  if (m_UseZeroAverageDisplacementConstraint)
  {
    if (!m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = m_GridSize[lastDim];
      const unsigned int numParametersPerDimension = this->GetNumberOfParameters() / MovingImageDimension;
      const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
      DerivativeType     mean(numControlPointsPerDimension);
      for (unsigned int d = 0; d < MovingImageDimension; ++d)
      {
        /** Compute mean per dimension. */
        mean.Fill(0.0);
        const unsigned int starti = numParametersPerDimension * d;
        for (unsigned int i = starti; i < starti + numParametersPerDimension; ++i)
        {
          mean[i % numControlPointsPerDimension] += derivative[i];
        }
        mean /= static_cast<double>(lastDimGridSize);

        /** Update derivative for every control point per dimension. */
        for (unsigned int i = starti; i < starti + numParametersPerDimension; ++i)
        {
          derivative[i] -= mean[i % numControlPointsPerDimension];
        }
      }
    }
    else
    {
      /** Update derivative per dimension.
       * Parameters are ordered x0x0x0y0y0y0z0z0z0x1x1x1y1y1y1z1z1z1 with
       * the number the time point index.
       */
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
      DerivativeType     mean(numParametersPerLastDimension, 0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          mean[c % numParametersPerLastDimension] += derivative[c];
        }
      }
      mean /= static_cast<double>(lastDimSize);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          derivative[c] -= mean[c % numParametersPerLastDimension];
        }
      }
    }
  }

  /** Return the measure value. */
  value = measure;

} // end GetValueAndDerivative()


} // end namespace itk

#endif // end #ifndef _itkVarianceOverLastDimensionImageMetric_hxx
