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
#ifndef itkPCAMetric_F_multithreaded_hxx
#define itkPCAMetric_F_multithreaded_hxx

#include "itkPCAMetric_F_multithreaded.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <vnl/algo/vnl_matrix_update.h>
#include "itkImage.h"
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_trace.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <numeric>
#include <fstream>

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
PCAMetric<TFixedImage, TMovingImage>::PCAMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

  /** Initialize the m_ParzenWindowHistogramThreaderParameters. */
  this->m_PCAMetricThreaderParameters.m_Metric = this;
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
  this->m_LastDimIndex = this->GetFixedImage()->GetImageDimension() - 1;
  this->m_G = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(m_LastDimIndex);

  if (this->m_NumEigenValues > this->m_G)
  {
    std::cerr << "ERROR: Number of eigenvalues is larger than number of images. Maximum number of eigenvalues equals: "
              << this->m_G << std::endl;
  }
} // end Initializes


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

} // end PrintSelf


/**
 * ********************* InitializeThreadingParameters ****************************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   * Filling the potentially large vectors is performed later, in each thread,
   * which has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  m_PCAMetricGetSamplesPerThreadVariables.resize(numberOfThreads);

  /** Some initialization. */
  for (auto & perThreadVariable : m_PCAMetricGetSamplesPerThreadVariables)
  {
    perThreadVariable.st_NumberOfPixelsCounted = NumericTraits<SizeValueType>::Zero;
    perThreadVariable.st_Derivative.SetSize(this->GetNumberOfParameters());
  }

  this->m_PixelStartIndex.resize(numberOfThreads);

} // end InitializeThreadingParameters()


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

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  /** Update the imageSampler and get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  const unsigned int numberOfSamples = sampleContainer->Size();
  MatrixType         datablock(numberOfSamples, this->m_G);

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
    for (unsigned int d = 0; d < this->m_G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[this->m_LastDimIndex] = d;

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

    if (numSamplesOk == this->m_G)
    {
      ++pixelIndex;
      this->m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(numberOfSamples, this->m_NumberOfPixelsCounted);
  MatrixType A(datablock.extract(this->m_NumberOfPixelsCounted, this->m_G));

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(this->m_G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < this->m_NumberOfPixelsCounted; ++i)
  {
    for (unsigned int j = 0; j < this->m_G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(this->m_NumberOfPixelsCounted);

  MatrixType Amm(this->m_NumberOfPixelsCounted, this->m_G);
  Amm.fill(NumericTraits<RealType>::Zero);

  for (unsigned int i = 0; i < this->m_NumberOfPixelsCounted; ++i)
  {
    for (unsigned int j = 0; j < this->m_G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  /** Compute covariance matrix C */
  MatrixType C(Amm.transpose() * Amm);
  C /= static_cast<RealType>(RealType(this->m_NumberOfPixelsCounted) - 1.0);

  vnl_diag_matrix<RealType> S(this->m_G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < this->m_G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  /** Compute correlation matrix K */
  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType sumEigenValuesUsed = itk::NumericTraits<RealType>::Zero;
  for (unsigned int i = 1; i < this->m_NumEigenValues + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(this->m_G - i);
  }

  measure = this->m_G - sumEigenValuesUsed;

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
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(const TransformParametersType & parameters,
                                                                          MeasureType &                   value,
                                                                          DerivativeType & derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::Zero);

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

  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  std::vector<FixedImagePointType> SamplesOK;

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  const unsigned int numberOfSamples = sampleContainer->Size();
  MatrixType         datablock(numberOfSamples, this->m_G);

  /** Initialize dummy loop variables */
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
    for (unsigned int d = 0; d < this->m_G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[this->m_LastDimIndex] = d;

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
    if (numSamplesOk == this->m_G)
    {
      SamplesOK.push_back(fixedPoint);
      ++pixelIndex;
      this->m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  MatrixType A(datablock.extract(this->m_NumberOfPixelsCounted, this->m_G));

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(this->m_G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < this->m_NumberOfPixelsCounted; ++i)
  {
    for (unsigned int j = 0; j < this->m_G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(this->m_NumberOfPixelsCounted);

  /** Calculate standard deviation from columns */
  MatrixType Amm(this->m_NumberOfPixelsCounted, this->m_G);
  Amm.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < this->m_NumberOfPixelsCounted; ++i)
  {
    for (unsigned int j = 0; j < this->m_G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  /** Compute covariance matrix C */
  MatrixType Atmm = Amm.transpose();
  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(this->m_NumberOfPixelsCounted) - 1.0);

  vnl_diag_matrix<RealType> S(this->m_G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < this->m_G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType sumEigenValuesUsed = itk::NumericTraits<RealType>::Zero;
  for (unsigned int i = 1; i < this->m_NumEigenValues + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(this->m_G - i);
  }

  MatrixType eigenVectorMatrix(this->m_G, this->m_NumEigenValues);
  for (unsigned int i = 1; i < this->m_NumEigenValues + 1; ++i)
  {
    eigenVectorMatrix.set_column(i - 1, (eig.get_eigenvector(this->m_G - i)).normalize());
  }

  MatrixType eigenVectorMatrixTranspose(eigenVectorMatrix.transpose());

  /** Create variables to store intermediate results in. */
  TransformJacobianType                   jacobian;
  DerivativeType                          dMTdmu;
  DerivativeType                          imageJacobian(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  std::vector<NonZeroJacobianIndicesType> nzjis(this->m_G, NonZeroJacobianIndicesType());

  /** Sub components of metric derivative */
  vnl_diag_matrix<DerivativeValueType> dSdmu_part1(this->m_G);

  /** initialize */
  dSdmu_part1.fill(itk::NumericTraits<DerivativeValueType>::Zero);

  for (unsigned int d = 0; d < this->m_G; ++d)
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
    FixedImagePointType fixedPoint = SamplesOK[pixelIndex];

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    for (unsigned int d = 0; d < this->m_G; ++d)
    {
      /** Initialize some variables. */
      RealType                  movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[this->m_LastDimIndex] = d;

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
        for (unsigned int z = 0; z < this->m_NumEigenValues; ++z)
        {
          derivative[nzjis[d][p]] += vSAtmm[z][pixelIndex] * dMTdmu[p] * Sv[d][z] +
                                     vdSdmu_part1[z][d] * Atmm[d][pixelIndex] * dMTdmu[p] * CSv[d][z];
        } // end loop over eigenvalues

      } // end loop over non-zero jacobian indices

    } // end loop over last dimension

  } // end second for loop over sample container

  derivative *= -(2.0 / (DerivativeValueType(this->m_NumberOfPixelsCounted) - 1.0)); // normalize
  measure = this->m_G - sumEigenValuesUsed;

  /** Subtract mean from derivative elements. */
  if (this->m_SubtractMean)
  {
    if (!this->m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = this->m_GridSize[this->m_LastDimIndex];
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
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->m_G;
      DerivativeType     mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < this->m_G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<RealType>(this->m_G);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < this->m_G; ++t)
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

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(const TransformParametersType & parameters,
                                                            MeasureType &                   value,
                                                            DerivativeType &                derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->GetValueAndDerivativeSingleThreaded(parameters, value, derivative);
  }

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

  this->InitializeThreadingParameters();

  /** Launch multi-threading GetSamples */
  this->LaunchGetSamplesThreaderCallback();

  /** Get the metric value contributions from all threads. */
  this->AfterThreadedGetSamples(value);

  /** Launch multi-threading ComputeDerivative */
  this->LaunchComputeDerivativeThreaderCallback();

  /** Sum derivative contributions from all threads */
  this->AfterThreadedComputeDerivative(derivative);

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetSamples *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::ThreadedGetSamples(ThreadIdType threadId)
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));
  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();
  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  std::vector<FixedImagePointType> SamplesOK;
  MatrixType                       datablock(nrOfSamplesPerThreads, this->m_G);

  unsigned int pixelIndex = 0;
  for (threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = threader_fiter->Value().m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
    for (unsigned int d = 0; d < this->m_G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[this->m_LastDimIndex] = d;

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);

      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      if (sampleOk)

      {
        sampleOk = this->FastEvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr, threadId);
      }

      if (sampleOk)
      {
        ++numSamplesOk;
        datablock(pixelIndex, d) = movingImageValue;
      } // end if sampleOk

    } // end loop over t
    if (numSamplesOk == m_G)
    {
      SamplesOK.push_back(fixedPoint);
      ++pixelIndex;
    }

  } /** end first loop over image sample container */

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_PCAMetricGetSamplesPerThreadVariables[threadId].st_NumberOfPixelsCounted = pixelIndex;
  this->m_PCAMetricGetSamplesPerThreadVariables[threadId].st_DataBlock = datablock.extract(pixelIndex, this->m_G);
  this->m_PCAMetricGetSamplesPerThreadVariables[threadId].st_ApprovedSamples = SamplesOK;

} // end ThreadedGetSamples()


/**
 * ******************* AfterThreadedGetSamples *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::AfterThreadedGetSamples(MeasureType & value) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_PCAMetricGetSamplesPerThreadVariables[0].st_NumberOfPixelsCounted;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    this->m_NumberOfPixelsCounted += this->m_PCAMetricGetSamplesPerThreadVariables[i].st_NumberOfPixelsCounted;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  MatrixType   A(this->m_NumberOfPixelsCounted, this->m_G);
  unsigned int row_start = 0;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    A.update(this->m_PCAMetricGetSamplesPerThreadVariables[i].st_DataBlock, row_start, 0);
    this->m_PixelStartIndex[i] = row_start;
    row_start += this->m_PCAMetricGetSamplesPerThreadVariables[i].st_DataBlock.rows();
  }

  /** Calculate mean of from columns */
  vnl_vector<RealType> mean(this->m_G);
  mean.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < this->m_NumberOfPixelsCounted; ++i)
  {
    for (unsigned int j = 0; j < this->m_G; ++j)
    {
      mean(j) += A(i, j);
    }
  }
  mean /= RealType(this->m_NumberOfPixelsCounted);

  /** Calculate standard deviation from columns */
  MatrixType Amm(this->m_NumberOfPixelsCounted, this->m_G);
  Amm.fill(NumericTraits<RealType>::Zero);
  for (unsigned int i = 0; i < this->m_NumberOfPixelsCounted; ++i)
  {
    for (unsigned int j = 0; j < this->m_G; ++j)
    {
      Amm(i, j) = A(i, j) - mean(j);
    }
  }

  /** Compute covariancematrix C */
  this->m_Atmm = Amm.transpose();
  MatrixType C(this->m_Atmm * Amm);
  C /= static_cast<RealType>(RealType(this->m_NumberOfPixelsCounted) - 1.0);

  vnl_diag_matrix<RealType> S(this->m_G);
  S.fill(NumericTraits<RealType>::Zero);
  for (unsigned int j = 0; j < this->m_G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType   sumEigenValuesUsed = itk::NumericTraits<RealType>::Zero;
  MatrixType eigenVectorMatrix(this->m_G, this->m_NumEigenValues);
  for (unsigned int i = 1; i < this->m_NumEigenValues + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(this->m_G - i);
    eigenVectorMatrix.set_column(i - 1, (eig.get_eigenvector(this->m_G - i)).normalize());
  }

  value = this->m_G - sumEigenValuesUsed;

  MatrixType eigenVectorMatrixTranspose(eigenVectorMatrix.transpose());

  /** Sub components of metric derivative */
  vnl_diag_matrix<DerivativeValueType> dSdmu_part1(this->m_G);

  for (unsigned int d = 0; d < this->m_G; ++d)
  {
    double S_sqr = S(d, d) * S(d, d);
    double S_qub = S_sqr * S(d, d);
    dSdmu_part1(d, d) = -S_qub;
  }

  this->m_vSAtmm = eigenVectorMatrixTranspose * S * this->m_Atmm;
  this->m_CSv = C * S * eigenVectorMatrix;
  this->m_Sv = S * eigenVectorMatrix;
  this->m_vdSdmu_part1 = eigenVectorMatrixTranspose * dSdmu_part1;

} // end AfterThreadedGetSamples()


/**
 * **************** GetSamplesThreaderCallback *******
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
PCAMetric<TFixedImage, TMovingImage>::GetSamplesThreaderCallback(void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadId = infoStruct->WorkUnitID;

  PCAMetricMultiThreaderParameterType * temp = static_cast<PCAMetricMultiThreaderParameterType *>(infoStruct->UserData);

  temp->m_Metric->ThreadedGetSamples(threadId);

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // GetSamplesThreaderCallback()


/**
 * *********************** LaunchGetSamplesThreaderCallback***************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::LaunchGetSamplesThreaderCallback() const
{
  /** Setup local threader. */
  // \todo: is a global threader better performance-wise? check
  auto local_threader = ThreaderType::New();
  local_threader->SetNumberOfWorkUnits(Self::GetNumberOfWorkUnits());
  local_threader->SetSingleMethod(this->GetSamplesThreaderCallback,
                                  const_cast<void *>(static_cast<const void *>(&this->m_PCAMetricThreaderParameters)));

  /** Launch. */
  local_threader->SingleMethodExecute();

} // end LaunchGetSamplesThreaderCallback()


/**
 * ******************* ThreadedComputeDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::ThreadedComputeDerivative(ThreadIdType threadId)
{
  /** Create variables to store intermediate results in. */
  DerivativeType & derivative = this->m_PCAMetricGetSamplesPerThreadVariables[threadId].st_Derivative;
  derivative.Fill(0.0);

  /** Initialize some variables. */
  RealType                  movingImageValue;
  MovingImageDerivativeType movingImageDerivative;

  TransformJacobianType      jacobian;
  DerivativeType             imageJacobian(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  NonZeroJacobianIndicesType nzjis(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());

  unsigned int dummyindex = 0;
  /** Second loop over fixed image samples. */
  for (unsigned int pixelIndex = this->m_PixelStartIndex[threadId];
       pixelIndex < (this->m_PixelStartIndex[threadId] +
                     this->m_PCAMetricGetSamplesPerThreadVariables[threadId].st_ApprovedSamples.size());
       ++pixelIndex)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint =
      this->m_PCAMetricGetSamplesPerThreadVariables[threadId].st_ApprovedSamples[dummyindex];

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    for (unsigned int d = 0; d < this->m_G; ++d)
    {
      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[this->m_LastDimIndex] = d;

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      this->FastEvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative, threadId);

      /** Get the TransformJacobian dT/dmu */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzjis);

      /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** build metric derivative components */
      for (unsigned int p = 0; p < nzjis.size(); ++p)
      {
        DerivativeValueType tmp = 0.0;
        for (unsigned int z = 0; z < this->m_NumEigenValues; ++z)
        {
          tmp += this->m_vSAtmm[z][pixelIndex] * imageJacobian[p] * this->m_Sv[d][z] +
                 this->m_vdSdmu_part1[z][d] * this->m_Atmm[d][pixelIndex] * imageJacobian[p] * this->m_CSv[d][z];
        } // end loop over eigenvalues
        derivative[nzjis[p]] += tmp;
      } // end loop over non-zero jacobian indices

    } // end loop over last dimension
    ++dummyindex;

  } // end second for loop over sample container

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedComputeDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::AfterThreadedComputeDerivative(DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  derivative = this->m_PCAMetricGetSamplesPerThreadVariables[0].st_Derivative;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    derivative += this->m_PCAMetricGetSamplesPerThreadVariables[i].st_Derivative;
  }

  derivative *= -(2.0 / (DerivativeValueType(this->m_NumberOfPixelsCounted) - 1.0)); // normalize

  /** Subtract mean from derivative elements. */
  if (this->m_SubtractMean)
  {
    if (!this->m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = this->m_GridSize[this->m_LastDimIndex];
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
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->m_G;
      DerivativeType     mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < this->m_G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<RealType>(this->m_G);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < this->m_G; ++t)
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
} // end AftherThreadedComputeDerivative()


/**
 * **************** ComputeDerivativeThreaderCallback *******
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
PCAMetric<TFixedImage, TMovingImage>::ComputeDerivativeThreaderCallback(void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadId = infoStruct->WorkUnitID;

  PCAMetricMultiThreaderParameterType * temp = static_cast<PCAMetricMultiThreaderParameterType *>(infoStruct->UserData);

  temp->m_Metric->ThreadedComputeDerivative(threadId);

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end omputeDerivativeThreaderCallback()


/**
 * ************** LaunchComputeDerivativeThreaderCallback **********
 */

template <class TFixedImage, class TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::LaunchComputeDerivativeThreaderCallback() const
{
  /** Setup local threader. */
  // \todo: is a global threader better performance-wise? check
  auto local_threader = ThreaderType::New();
  local_threader->SetNumberOfWorkUnits(Self::GetNumberOfWorkUnits());
  local_threader->SetSingleMethod(this->ComputeDerivativeThreaderCallback,
                                  const_cast<void *>(static_cast<const void *>(&this->m_PCAMetricThreaderParameters)));

  /** Launch. */
  local_threader->SingleMethodExecute();

} // end LaunchComputeDerivativeThreaderCallback()


} // end namespace itk

#endif // itkPCAMetric_F_multithreaded_hxx
