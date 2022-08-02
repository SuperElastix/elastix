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
#ifndef itkPCAMetric2_hxx
#define itkPCAMetric2_hxx

#include "itkPCAMetric2.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <vnl/algo/vnl_matrix_update.h>
#include "itkImage.h"
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_trace.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <numeric>
#include <fstream>

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
PCAMetric2<TFixedImage, TMovingImage>::PCAMetric2()
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
PCAMetric2<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Retrieve slowest varying dimension and its size. */
  // const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  // const unsigned int G = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric2<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* SampleRandom *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric2<TFixedImage, TMovingImage>::SampleRandom(const int n, const int m, std::vector<int> & numbers) const
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
PCAMetric2<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
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
} // end EvaluateTransformJacobianInnerProduct()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
auto
PCAMetric2<TFixedImage, TMovingImage>::GetValue(const TransformParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");
  bool UseGetValueAndDerivative = false;

  if (UseGetValueAndDerivative)
  {
    using DerivativeValueType = typename DerivativeType::ValueType;
    const unsigned int numberOfParameters = this->GetNumberOfParameters();
    MeasureType        dummymeasure = NumericTraits<MeasureType>::Zero;
    DerivativeType     dummyderivative = DerivativeType(numberOfParameters);
    dummyderivative.Fill(NumericTraits<DerivativeValueType>::Zero);

    this->GetValueAndDerivative(parameters, dummymeasure, dummyderivative);
    return dummymeasure;
  }

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
  const unsigned int numberOfSamples = sampleContainer->Size();
  MatrixType         datablock(numberOfSamples, G);

  /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
  std::vector<int> lastDimPositions;

  /** Determine random last dimension positions if needed. */
  for (unsigned int i = 0; i < G; ++i)
  {
    lastDimPositions.push_back(i);
  }

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
      voxelCoord[lastDim] = lastDimPositions[d];

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
  this->CheckNumberOfSamples(numberOfSamples, this->m_NumberOfPixelsCounted);
  const unsigned int N = this->m_NumberOfPixelsCounted;
  MatrixType         A(datablock.extract(N, G));

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

  /** Compute covariancematrix C */
  MatrixType C(Amm.transpose() * Amm);
  C /= static_cast<RealType>(RealType(N) - 1.0);

  MatrixType S(G, G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  /** Compute correlation matrix K */
  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  // The measure is the sum of weighted eigenvalues of the correlation matrix.
  // measure = sum_{i=1}^G i*lambda_i

  // Note that the system does NOT crash when you violate the number of possible
  // eigenvalues, i.e. when K is of size 30x30, eigenvalues > 30 also exist and have
  // a value.

  // The eigenvalues of vnl_symmetric_eigensystem are in ascending order, meaning that
  // when K is of size 30x30, eigenvalue 29 is the highest, and eigenvalue 0 is the lowest.
  // We want the low eigenvalue to get the highest weight and the highest eigenvalue to get
  // the lowest weight, i.e. for K of size 30x30:
  // eigenvalue 29 has a weight of 1 and eigenvalue 0 has a weight of 30

  RealType sumWeightedEigenValues = itk::NumericTraits<RealType>::Zero;
  for (unsigned int i = 0; i < G; ++i)
  {
    sumWeightedEigenValues += (i + 1) * eig.get_eigenvalue(G - i - 1);
  }

  measure = sumWeightedEigenValues;

  /** Return the measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric2<TFixedImage, TMovingImage>::GetDerivative(const TransformParametersType & parameters,
                                                     DerivativeType &                derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable. */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;

  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric2<TFixedImage, TMovingImage>::GetValueAndDerivative(const TransformParametersType & parameters,
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
  const unsigned int numberOfSamples = sampleContainer->Size();
  MatrixType         datablock(numberOfSamples, G);

  /** Initialize dummy loop variables */
  unsigned int pixelIndex = 0;

  /** Initialize image sample matrix . */
  datablock.fill(itk::NumericTraits<RealType>::Zero);

  /** Determine random last dimension positions if needed. */
  /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
  /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
  std::vector<int> lastDimPositions;

  for (unsigned int i = 0; i < G; ++i)
  {
    lastDimPositions.push_back(i);
  }

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
      voxelCoord[lastDim] = lastDimPositions[d];

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

    } // end loop over gradient images
    if (numSamplesOk == G)
    {
      SamplesOK.push_back(fixedPoint);
      ++pixelIndex;
      this->m_NumberOfPixelsCounted++;
    }
  }

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);
  unsigned int N = this->m_NumberOfPixelsCounted;

  MatrixType A(datablock.extract(N, G));

  /** Calculate mean of columns */
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

  /** Calculate standard deviation of columns */
  MatrixType Amm(N, G);
  Amm.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < N; ++i)
  {
    for (unsigned int j = 0; j < G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  /** Compute covariance matrix C */
  MatrixType Atmm = Amm.transpose();
  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(N) - 1.0);

  vnl_diag_matrix<RealType> S(G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  /** Compute correlation matrix K */
  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType sumWeightedEigenValues = itk::NumericTraits<RealType>::Zero;
  for (unsigned int i = 0; i < G; ++i)
  {
    sumWeightedEigenValues += (i + 1) * eig.get_eigenvalue(G - i - 1);
  }

  MatrixType eigenVectorMatrix(G, G);
  for (unsigned int i = 0; i < G; ++i)
  {
    eigenVectorMatrix.set_column(i, (eig.get_eigenvector(G - i - 1)).normalize());
  }

  MatrixType eigenVectorMatrixTranspose(eigenVectorMatrix.transpose());

  /** Create variables to store intermediate results in. */
  TransformJacobianType                   jacobian;
  DerivativeType                          dMTdmu;
  DerivativeType                          imageJacobian(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  std::vector<NonZeroJacobianIndicesType> nzjis(G, NonZeroJacobianIndicesType());

  /** Sub components of metric derivative */
  vnl_diag_matrix<DerivativeValueType> dSdmu_part1(G);

  unsigned int startSamplesOK;
  startSamplesOK = 0;

  for (unsigned int d = 0; d < G; ++d)
  {
    double S_sqr = S(d, d) * S(d, d);
    double S_qub = S_sqr * S(d, d);
    dSdmu_part1(d, d) = -S_qub;
  }

  DerivativeMatrixType vSAtmm(eigenVectorMatrixTranspose * S * Atmm);
  DerivativeMatrixType CSv(C * S * eigenVectorMatrix);
  DerivativeMatrixType Sv(S * eigenVectorMatrix);
  DerivativeMatrixType vdSdmu_part1(eigenVectorMatrixTranspose * dSdmu_part1);

  /** Second loop over fixed image samples. */
  for (pixelIndex = 0; pixelIndex < SamplesOK.size(); ++pixelIndex)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = SamplesOK[startSamplesOK];
    ++startSamplesOK;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    const unsigned int G = lastDimPositions.size();

    for (unsigned int d = 0; d < G; ++d)
    {
      /** Initialize some variables. */
      RealType                  movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = lastDimPositions[d];

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
        for (unsigned int z = 0; z < G; ++z)
        {
          derivative[nzjis[d][p]] += z * (vSAtmm[z][pixelIndex] * dMTdmu[p] * Sv[d][z] +
                                          vdSdmu_part1[z][d] * Atmm[d][pixelIndex] * dMTdmu[p] * CSv[d][z]);
        } // end loop over eigenvalues

      } // end loop over non-zero jacobian indices

    } // end loop over last dimension

  } // end second for loop over sample container

  derivative *= (2.0 / (DerivativeValueType(N) - 1.0)); // normalize
  measure = sumWeightedEigenValues;

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
        mean /= static_cast<RealType>(lastDimGridSize);

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
      mean /= static_cast<RealType>(G);

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

#endif // itkPCAMetric2_hxx
