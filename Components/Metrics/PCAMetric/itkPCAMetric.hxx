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
#ifndef itkPCAMetric_hxx
#define itkPCAMetric_hxx

#include "itkPCAMetric.h"

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
PCAMetric<TFixedImage, TMovingImage>::PCAMetric()
  : m_SubtractMean(false)
  , m_TransformIsStackTransform(false)
  , m_NumEigenValues(6)
  , m_UseDerivativeOfMean(false)
  , m_DeNoise(false)
  , m_VarNoise(0.0)
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
PCAMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Check num last samples. */
  if (this->m_NumSamplesLastDimension > lastDimSize)
  {
    this->m_NumSamplesLastDimension = lastDimSize;
  }

  if (this->m_NumEigenValues > lastDimSize)
  {
    std::cout << "ERROR: Number of eigenvalues is larger than number of images. Maximum number of eigenvalues equals: "
              << lastDimSize << std::endl;
  }
} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* SampleRandom *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::SampleRandom(const int n, const int m, std::vector<int> & numbers) const
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
PCAMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
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
PCAMetric<TFixedImage, TMovingImage>::GetValue(const TransformParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");
  bool UseGetValueAndDerivative = false;

  if (UseGetValueAndDerivative)
  {
    using DerivativeValueType = typename DerivativeType::ValueType;
    const unsigned int P = this->GetNumberOfParameters();
    MeasureType        dummymeasure = NumericTraits<MeasureType>::Zero;
    DerivativeType     dummyderivative = DerivativeType(P);
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
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);
  const unsigned int numLastDimSamples = this->m_NumSamplesLastDimension;

  using MatrixType = vnl_matrix<RealType>;

  /** Get real last dim samples. */
  const unsigned int realNumLastDimPositions = this->m_SampleLastDimensionRandomly
                                                 ? this->m_NumSamplesLastDimension + this->m_NumAdditionalSamplesFixed
                                                 : lastDimSize;

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  unsigned int NumberOfSamples = sampleContainer->Size();
  MatrixType   datablock(NumberOfSamples, realNumLastDimPositions);

  /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
  std::vector<int> lastDimPositions;

  /** Determine random last dimension positions if needed. */

  if (this->m_SampleLastDimensionRandomly)
  {
    SampleRandom(this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions);
  }
  else
  {
    for (unsigned int i = 0; i < lastDimSize; ++i)
    {
      lastDimPositions.push_back(i);
    }
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
    FixedImageContinuousIndexType voxelCoord;
    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex(fixedPoint, voxelCoord);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
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

      if (sampleOk)
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, 0);
      }

      if (sampleOk)
      {
        ++numSamplesOk;
        datablock(pixelIndex, d) = movingImageValue;
      }

    } /** end loop over t */

    if (numSamplesOk == realNumLastDimPositions)
    {
      ++pixelIndex;
      this->m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(NumberOfSamples, this->m_NumberOfPixelsCounted);
  unsigned int N = this->m_NumberOfPixelsCounted;
  this->m_NumberOfSamples = N;
  const unsigned int G = realNumLastDimPositions;
  MatrixType         A(datablock.extract(N, G));

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(N);

  MatrixType Amm(N, G);
  Amm.fill(NumericTraits<RealType>::Zero);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  /** Transpose of the matrix with mean subtracted */
  MatrixType Atmm(Amm.transpose());

  /** Compute covariancematrix C */
  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(N) - 1.0);

  vnl_symmetric_eigensystem<RealType> eigc(C);

  RealType varNoise = 0.9999999 * eigc.get_eigenvalue(0);

  if (!this->m_DeNoise)
  {
    varNoise = this->m_VarNoise;
  }
  // std::cout << "varNoise: " << varNoise << std::endl;

  /** Calculate variance of columns */
  vnl_vector<RealType> var(G);
  var.fill(NumericTraits<RealType>::Zero);
  for (int j = 0; j < G; ++j)
  {
    var(j) = C(j, j);
  }
  var -= varNoise;

  MatrixType S(G, G);
  S.fill(NumericTraits<RealType>::Zero);
  for (int j = 0; j < G; ++j)
  {
    S(j, j) = 1.0 / sqrt(var(j));
  }

  for (int j = 0; j < G; ++j)
  {
    C(j, j) -= varNoise;
  }

  /** Compute correlation matrix K */
  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  /** Compute sum of all eigenvalues = trace( K ) */
  //    RealType trace = itk::NumericTraits< RealType >::Zero;
  //    for( int i = 0; i < G; i++ )
  //    {
  //        trace += K(i,i);
  //    }

  //    RealType trace = vnl_trace( K );
  const unsigned int L = this->m_NumEigenValues;

  RealType sumEigenValuesUsed = itk::NumericTraits<RealType>::Zero;
  for (unsigned int i = 1; i < L + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(G - i);
  }

  //    measure = trace - sumEigenValuesUsed;
  measure = G - sumEigenValuesUsed;

  /** Return the measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::GetDerivative(const TransformParametersType & parameters,
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
PCAMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(const TransformParametersType & parameters,
                                                            MeasureType &                   value,
                                                            DerivativeType &                derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");
  /** Define derivative and Jacobian types. */
  using DerivativeValueType = typename DerivativeType::ValueType;
  using TransformJacobianValueType = typename TransformJacobianType::ValueType;

  /** Initialize some variables */
  const unsigned int P = this->GetNumberOfParameters();
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(P);
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
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);
  const unsigned int numLastDimSamples = this->m_NumSamplesLastDimension;

  using MatrixType = vnl_matrix<RealType>;
  using DerivativeMatrixType = vnl_matrix<DerivativeValueType>;

  std::vector<FixedImagePointType> SamplesOK;

  /** Get real last dim samples. */
  const unsigned int realNumLastDimPositions = this->m_SampleLastDimensionRandomly
                                                 ? this->m_NumSamplesLastDimension + this->m_NumAdditionalSamplesFixed
                                                 : lastDimSize;
  const unsigned int G = realNumLastDimPositions;

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  unsigned int NumberOfSamples = sampleContainer->Size();
  MatrixType   datablock(NumberOfSamples, G);

  /** Initialize dummy loop variables */
  unsigned int pixelIndex = 0;

  /** Initialize image sample matrix . */
  datablock.fill(itk::NumericTraits<RealType>::Zero);

  /** Determine random last dimension positions if needed. */
  /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
  std::vector<int> lastDimPositions;
  if (this->m_SampleLastDimensionRandomly)
  {
    SampleRandom(this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions);
  }
  else
  {
    for (unsigned int i = 0; i < lastDimSize; ++i)
    {
      lastDimPositions.push_back(i);
    }
  }

  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    FixedImageContinuousIndexType voxelCoord;
    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex(fixedPoint, voxelCoord);

    const unsigned int G = lastDimPositions.size();
    unsigned int       numSamplesOk = 0;

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
        sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, 0);
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
  this->m_NumberOfSamples = this->m_NumberOfPixelsCounted;

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);
  unsigned int N = pixelIndex;

  MatrixType A(datablock.extract(N, G));

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(N);

  /** Calculate standard deviation from columns */
  MatrixType Amm(N, G);
  Amm.fill(NumericTraits<RealType>::Zero);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  MatrixType Atmm(Amm.transpose());

  /** Compute covariancematrix C */
  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(N) - 1.0);

  vnl_symmetric_eigensystem<RealType> eigc(C);
  vnl_vector<RealType>                v_G = eigc.get_eigenvector(0);
  RealType                            varNoise = 0.9999999 * eigc.get_eigenvalue(0);
  if (!this->m_DeNoise)
  {
    varNoise = this->m_VarNoise;
  }

  /** Calculate standard deviation from columns */
  vnl_vector<RealType> var(G);
  var.fill(NumericTraits<RealType>::Zero);
  for (int j = 0; j < G; ++j)
  {
    var(j) = C(j, j);
  }
  var -= varNoise;

  vnl_diag_matrix<RealType> S(G);
  S.fill(NumericTraits<RealType>::Zero);
  for (int j = 0; j < G; ++j)
  {
    S(j, j) = 1.0 / sqrt(var(j));
  }

  for (int j = 0; j < G; ++j)
  {
    C(j, j) -= varNoise;
  }

  /** Compute correlation matrix K */
  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  //    /** Compute sum of all eigenvalues = trace( K ) */
  //    RealType trace = itk::NumericTraits< RealType >::Zero;
  //    for( int i = 0; i < G; i++ )
  //    {
  //        trace += K(i,i);
  //    }

  const unsigned int L = this->m_NumEigenValues;

  RealType sumEigenValuesUsed = itk::NumericTraits<RealType>::Zero;
  for (unsigned int i = 1; i < L + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(G - i);
  }

  MatrixType eigenVectorMatrix(G, L);
  for (unsigned int i = 1; i < L + 1; ++i)
  {
    eigenVectorMatrix.set_column(i - 1, (eig.get_eigenvector(G - i)).normalize());
  }

  MatrixType eigenVectorMatrixTranspose(eigenVectorMatrix.transpose());

  /** Create variables to store intermediate results in. */
  TransformJacobianType                   jacobian;
  DerivativeType                          dMTdmu;
  DerivativeType                          imageJacobian(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  std::vector<NonZeroJacobianIndicesType> nzjis(G, NonZeroJacobianIndicesType());

  /** Sub components of metric derivative */
  vnl_vector<DerivativeValueType> tracevKvdmu(P);
  vnl_vector<DerivativeValueType> tracevdSdmuCSv(P);
  vnl_vector<DerivativeValueType> tracevSdCdmuSv(P);
  vnl_vector<DerivativeValueType> tracevSdvarNoisedmuSv(P);
  vnl_vector<DerivativeValueType> dSdmu_part1(G);
  vnl_vector<DerivativeValueType> dvarNoisedmu(P);

  DerivativeMatrixType vSAtmmdAdmu(L, G * P);
  DerivativeMatrixType vAtmmdAdmu(L, G * P);
  DerivativeMatrixType dSdmu(G, P);
  DerivativeMatrixType v_GAtmmdAdmu(G, P);

  /** initialize */
  vSAtmmdAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  tracevKvdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  v_GAtmmdAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  tracevdSdmuCSv.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  tracevSdCdmuSv.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  tracevSdvarNoisedmuSv.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  dvarNoisedmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  vAtmmdAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  dSdmu_part1.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  dSdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);

  /** Components for derivative of mean */
  vnl_vector<DerivativeValueType> meandAdmu(P);
  DerivativeMatrixType            meandSdmu(G, P);
  DerivativeMatrixType            v_GAtmmmeandAdmu(G, P);
  DerivativeMatrixType            vAtmmmeandAdmu(L, G * P);
  DerivativeMatrixType            vSAtmmmeandAdmu(L, G * P);
  meandAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  v_GAtmmmeandAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  vSAtmmmeandAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  vAtmmmeandAdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  meandSdmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);

  unsigned int startSamplesOK;
  startSamplesOK = 0;

  for (unsigned int d = 0; d < G; ++d)
  {
    dSdmu_part1[d] = -pow(S(d, d), 3.0);
  }

  DerivativeMatrixType            vSAtmm(eigenVectorMatrixTranspose * S * Atmm);
  DerivativeMatrixType            CSv(C * S * eigenVectorMatrix);
  DerivativeMatrixType            Sv(S * eigenVectorMatrix);
  DerivativeMatrixType            vS(eigenVectorMatrixTranspose * S);
  vnl_vector<DerivativeValueType> v_GAtmm(v_G * Atmm);

  /** Second loop over fixed image samples. */
  for (pixelIndex = 0; pixelIndex < SamplesOK.size(); ++pixelIndex)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = SamplesOK[startSamplesOK];
    ++startSamplesOK;

    /** Transform sampled point to voxel coordinates. */
    FixedImageContinuousIndexType voxelCoord;
    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex(fixedPoint, voxelCoord);

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

      this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative);

      /** Get the TransformJacobian dT/dmu */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzjis[d]);

      /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Store values. */
      dMTdmu = imageJacobian;
      /** build metric derivative components */
      for (unsigned int p = 0; p < nzjis[d].size(); ++p)
      {
        meandAdmu[nzjis[d][p]] += (dMTdmu[p]) / double(N);
        v_GAtmmdAdmu[d][nzjis[d][p]] += v_GAtmm[pixelIndex] * dMTdmu[p];
        dSdmu[d][nzjis[d][p]] += Atmm[d][pixelIndex] * dSdmu_part1[d] * dMTdmu[p];
        for (unsigned int z = 0; z < L; ++z)
        {
          vSAtmmdAdmu[z][d + nzjis[d][p] * G] += vSAtmm[z][pixelIndex] * dMTdmu[p];
        }
      }
    } // end loop over t
  }   // end second for loop over sample container

  if (this->m_UseDerivativeOfMean)
  {
    for (unsigned int i = 0; i < N; ++i)
    {
      for (unsigned int d = 0; d < G; ++d)
      {
        for (unsigned int p = 0; p < P; ++p)
        {
          v_GAtmmmeandAdmu[d][p] += v_GAtmm[i] * meandAdmu[p];
          meandSdmu[d][p] += Atmm[d][i] * dSdmu_part1[d] * meandAdmu[p];
          for (unsigned int z = 0; z < L; ++z)
          {
            vSAtmmmeandAdmu[z][d + G * p] += vSAtmm[z][i] * meandAdmu[p];
          }
        }
      }
    }
  }

  for (unsigned int p = 0; p < P; ++p)
  {
    dvarNoisedmu[p] = dot_product((v_GAtmmdAdmu - v_GAtmmmeandAdmu).get_column(p), v_G);
  }

  if (!this->m_DeNoise)
  {
    dvarNoisedmu.fill(itk::NumericTraits<DerivativeValueType>::Zero);
  }

  for (unsigned int p = 0; p < P; ++p)
  {
    vnl_diag_matrix<DerivativeValueType> diagonaldvarNoisedmu(G);
    diagonaldvarNoisedmu.fill(0.0);
    vnl_diag_matrix<DerivativeValueType> diagonaldSdmu(G);
    diagonaldSdmu.fill(0.0);
    for (unsigned int d = 0; d < G; ++d)
    {
      diagonaldvarNoisedmu(d, d) = dSdmu_part1[d] * dvarNoisedmu[p];
      diagonaldSdmu(d, d) = (dSdmu - meandSdmu).get_column(p)[d] - diagonaldvarNoisedmu(d, d);
    }
    DerivativeMatrixType vSdCdmuSv = ((vSAtmmdAdmu - vSAtmmmeandAdmu).extract(L, G, 0, p * G)) * Sv;
    DerivativeMatrixType vSdvarNoisedmuSv = (vS * dvarNoisedmu[p] * Sv);

    tracevSdCdmuSv[p] = vnl_trace<DerivativeValueType>(vSdCdmuSv);
    tracevSdvarNoisedmuSv[p] = vnl_trace<DerivativeValueType>(vSdvarNoisedmuSv);
    tracevdSdmuCSv[p] = vnl_trace<DerivativeValueType>(eigenVectorMatrixTranspose * diagonaldSdmu * CSv);
  }

  tracevKvdmu = tracevdSdmuCSv + tracevSdCdmuSv - tracevSdvarNoisedmuSv;
  tracevKvdmu *= (2.0 / (DerivativeValueType(N) - 1.0)); // normalize

  // measure = trace - sumEigenValuesUsed;
  measure = G - sumEigenValuesUsed;
  derivative = -tracevKvdmu;

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
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
      DerivativeType     mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<RealType>(lastDimSize);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
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

  //    /** Compute norm of transform parameters per image */
  //    this->m_normdCdmu.set_size(lastDimSize);
  //    this->m_normdCdmu.fill(0.0);
  //    unsigned int ind = 0;
  //    for ( unsigned int t = 0; t < lastDimSize; ++t )
  //    {
  //        const unsigned int startc = (this->GetNumberOfParameters() / lastDimSize)*t;
  //        for ( unsigned int c = startc; c < startc + (this->GetNumberOfParameters() / lastDimSize); ++c )
  //        {
  //         this->m_normdCdmu[ ind ] += pow(derivative[ c ],2);
  //        }
  //        ++ind;
  //    }

  //    for(unsigned int index = 0; index < this->m_normdCdmu.size(); index++)
  //    {
  //        this->m_normdCdmu[index] = sqrt(this->m_normdCdmu.get(index));
  //    }

  //    this->m_normdCdmu /= static_cast< RealType >( this->GetNumberOfParameters() / lastDimSize );

  /** Return the measure value. */
  value = measure;

} // end GetValueAndDerivative()


} // end namespace itk

#endif // itkPCAMetric_hxx
