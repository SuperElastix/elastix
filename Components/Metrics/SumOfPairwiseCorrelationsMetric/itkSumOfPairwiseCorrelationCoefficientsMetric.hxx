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
#ifndef itkSumOfPairwiseCorrelationCoefficientsMetric_hxx
#define itkSumOfPairwiseCorrelationCoefficientsMetric_hxx

#include "itkSumOfPairwiseCorrelationCoefficientsMetric.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <vnl/algo/vnl_matrix_update.h>
#include "itkImage.h"
#include <numeric>

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::SumOfPairwiseCorrelationCoefficientsMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);
} // end constructor


/**
 * ******************* Initialize *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();
} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* SampleRandom *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::SampleRandom(const int          n,
                                                                                    const int          m,
                                                                                    std::vector<int> & numbers) const
{
  /** Empty list of last dimension positions. */
  numbers.clear();

  /** Initialize random number generator. */
  Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator =
    Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();

  /** Sample additional at fixed timepoint. */
  for (unsigned int i = 0; i < m_NumAdditionalSamplesFixed; ++i)
  {
    numbers.push_back(this->m_ReducedDimensionIndex);
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

template <class TFixedImage, class TMovingImage>
void
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  using JacobianIteratorType = typename TransformJacobianType::const_iterator;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill(0.0);

  for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
  {
    const double imDeriv = movingImageDerivative[dim];

    for (auto & imageJacobianElement : imageJacobian)
    {
      imageJacobianElement += (*jac) * imDeriv;
      ++jac;
    }
  }
} // end EvaluateTransformJacobianInnerProduct


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
auto
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int G = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  using MatrixType = vnl_matrix<RealType>;

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  unsigned int NumberOfSamples = sampleContainer->Size();
  MatrixType   datablock(NumberOfSamples, G);

  /** Initialize dummy loop variable */
  unsigned int pixelIndex = 0;

  /** Initialize image sample matrix . */
  datablock.fill(itk::NumericTraits<RealType>::Zero);

  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
    for (unsigned int d = 0; d < G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = d;

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);

      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      if (sampleOk)
      {
        sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
      }

      if (sampleOk)
      {
        ++numSamplesOk;
        datablock(pixelIndex, d) = movingImageValue;
      }

    } /** end loop over t */

    if (numSamplesOk == G)
    {
      ++pixelIndex;
      this->m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(NumberOfSamples, this->m_NumberOfPixelsCounted);
  unsigned int N = this->m_NumberOfPixelsCounted;

  MatrixType A(datablock.extract(N, G));

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < N; ++i)
  {
    for (unsigned int j = 0; j < G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(N);

  MatrixType Amm(N, G);
  Amm.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < N; ++i)
  {
    for (unsigned int j = 0; j < G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  MatrixType Atmm = Amm.transpose();

  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(N) - 1.0);

  vnl_diag_matrix<RealType> S(G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  MatrixType K(S * C * S);

  measure = 1.0 - (K.fro_norm() / RealType(G));

  /** Return the measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType &                derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;

  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumOfPairwiseCorrelationCoefficientsMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Define derivative and Jacobian types. */
  using DerivativeValueType = typename DerivativeType::ValueType;
  // typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

  /** Initialize some variables */
  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(numberOfParameters);
  derivative.Fill(NumericTraits<DerivativeValueType>::Zero);

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int G = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  using MatrixType = vnl_matrix<RealType>;
  using DerivativeMatrixType = vnl_matrix<DerivativeValueType>;

  std::vector<FixedImagePointType> SamplesOK;

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  unsigned int NumberOfSamples = sampleContainer->Size();
  MatrixType   datablock(NumberOfSamples, G);

  /** Initialize dummy loop variables */
  unsigned int pixelIndex = 0;

  /** Initialize image sample matrix . */
  datablock.fill(0.0);

  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
    for (unsigned int d = 0; d < G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = d;

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);

      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      if (sampleOk)
      {
        sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
      }

      if (sampleOk)
      {
        ++numSamplesOk;
        datablock(pixelIndex, d) = movingImageValue;

      } // end if sampleOk

    } // end loop over t

    if (numSamplesOk == G)
    {
      SamplesOK.push_back(fixedPoint);
      ++pixelIndex;
      this->m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);
  unsigned int N = this->m_NumberOfPixelsCounted;

  MatrixType A(datablock.extract(N, G));

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < N; ++i)
  {
    for (unsigned int j = 0; j < G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(N);

  MatrixType Amm(N, G);
  Amm.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < N; ++i)
  {
    for (unsigned int j = 0; j < G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  MatrixType Atmm = Amm.transpose();

  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(N) - 1.0);

  vnl_diag_matrix<RealType> S(G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  DerivativeMatrixType K(S * C * S);

  /** Create variables to store intermediate results in. */
  TransformJacobianType                   jacobian;
  DerivativeType                          imageJacobian(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  std::vector<NonZeroJacobianIndicesType> nzjis(G, NonZeroJacobianIndicesType());

  DerivativeType dMTdmu;

  /** Sub components of metric derivative */
  vnl_diag_matrix<DerivativeValueType> dSdmu_part1(G);

  /** initialize */
  dSdmu_part1.fill(itk::NumericTraits<DerivativeValueType>::Zero);

  for (unsigned int d = 0; d < G; ++d)
  {
    double S_sqr = S(d, d) * S(d, d);
    double S_qub = S_sqr * S(d, d);
    dSdmu_part1(d, d) = -S_qub / (DerivativeValueType(N) - 1.0);
  }

  // DerivativeMatrixType CS( C*S );
  DerivativeMatrixType KAtZscore(K * (Amm * S).transpose());
  DerivativeMatrixType KAtZscoreAmm(K * (Amm * S).transpose() * Amm);

  /** Second loop over fixed image samples. */
  for (pixelIndex = 0; pixelIndex < SamplesOK.size(); ++pixelIndex)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = SamplesOK[pixelIndex];

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    for (unsigned int d = 0; d < G; ++d)
    {
      /** Initialize some variables. */
      RealType                  movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = d;

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative);

      /** Get the TransformJacobian dT/dmu */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzjis[d]);

      /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Store values. */
      dMTdmu = imageJacobian;

      /** build metric derivative components */
      for (unsigned int p = 0; p < nzjis[d].size(); ++p)
      {
        derivative[nzjis[d][p]] += KAtZscore[d][pixelIndex] * dMTdmu[p] * S(d, d);
        derivative[nzjis[d][p]] += dSdmu_part1(d, d) * Atmm[d][pixelIndex] * dMTdmu[p] * KAtZscoreAmm[d][d];
      } // end loop over non-zero jacobian indices

    } // end loop over t

  } // end second for loop over sample container

  derivative *= -static_cast<DerivativeValueType>(2.0) /
                (static_cast<DerivativeValueType>(N - static_cast<DerivativeValueType>(1.0)) *
                 (K.fro_norm() * RealType(G))); // normalize

  measure = RealType(1.0 - (K.fro_norm() / RealType(G)));

  /** Subtract mean from derivative elements. */
  if (this->m_SubtractMean)
  {
    if (!this->m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = this->m_GridSize[lastDim];
      const unsigned int numParametersPerDimension =
        this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
      const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
      DerivativeType     mean(numControlPointsPerDimension);
      for (unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d)
      {
        /** Compute mean per dimension. */
        mean.Fill(0.0);
        const unsigned int starti = numParametersPerDimension * d;
        for (unsigned int i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned int index = i % numControlPointsPerDimension;
          mean[index] += derivative[i];
        }
        mean /= static_cast<double>(lastDimGridSize);

        /** Update derivative for every control point per dimension. */
        for (unsigned int i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned int index = i % numControlPointsPerDimension;
          derivative[i] -= mean[index];
        }
      }
    }
    else
    {
      /** Update derivative per dimension.
       * Parameters are ordered x0x0x0y0y0y0z0z0z0x1x1x1y1y1y1z1z1z1 with
       * the number the time point index.
       */
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / G;
      DerivativeType     mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<double>(G);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          derivative[c] -= mean[index];
        }
      }
    }
  }

  /** Return the measure value. */
  value = measure;

} // end GetValueAndDerivative()


} // end namespace itk

#endif // itkSumOfPairwiseCorrelationCoefficientsMetric_hxx
