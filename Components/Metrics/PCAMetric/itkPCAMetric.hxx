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

#include <vnl/algo/vnl_matrix_update.h>
#include "itkImage.h"
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_trace.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <cassert>
#include <numeric>
#include <fstream>

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <typename TFixedImage, typename TMovingImage>
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

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Retrieve slowest varying dimension and its size. */
  m_LastDimIndex = FixedImageDimension - 1;
  m_G = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(m_LastDimIndex);

  if (m_NumEigenValues > m_G)
  {
    std::cerr << "ERROR: Number of eigenvalues is larger than number of images. Maximum number of eigenvalues equals: "
              << m_G << std::endl;
  }
} // end Initializes


/**
 * ******************* PrintSelf *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

} // end PrintSelf


/**
 * ********************* InitializeThreadingParameters ****************************
 */

template <typename TFixedImage, typename TMovingImage>
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
    perThreadVariable.st_NumberOfPixelsCounted = SizeValueType{};
    perThreadVariable.st_Derivative.SetSize(this->GetNumberOfParameters());
  }

  m_PixelStartIndex.resize(numberOfThreads);

} // end InitializeThreadingParameters()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  ImplementationDetails::EvaluateInnerProduct(jacobian, movingImageDerivative, imageJacobian);
}

/**
 * ******************* GetValue *******************
 */

template <typename TFixedImage, typename TMovingImage>
auto
PCAMetric<TFixedImage, TMovingImage>::GetValue(const ParametersType & parameters) const -> MeasureType
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
  Superclass::m_NumberOfPixelsCounted = 0;
  MeasureType measure{};

  /** Update the imageSampler and get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  const size_t numberOfSamples{ sampleContainer->size() };
  MatrixType   datablock(numberOfSamples, m_G, vnl_matrix_null);

  /** Initialize dummy loop variable */
  unsigned int pixelIndex = 0;

  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fixedImageSample.m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
    for (unsigned int d = 0; d < m_G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[m_LastDimIndex] = d;

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

    if (numSamplesOk == m_G)
    {
      ++pixelIndex;
      Superclass::m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();
  MatrixType A(datablock.extract(Superclass::m_NumberOfPixelsCounted, m_G));

  MatrixType Amm(Superclass::m_NumberOfPixelsCounted, m_G, vnl_matrix_null);
  {
    /** Calculate mean of from columns */
    vnl_vector<RealType> mean(m_G, RealType{});
    for (unsigned int i = 0; i < Superclass::m_NumberOfPixelsCounted; ++i)
    {
      for (unsigned int j = 0; j < m_G; ++j)
      {
        mean(j) += A(i, j);
      }
    }
    mean /= RealType(Superclass::m_NumberOfPixelsCounted);

    for (unsigned int i = 0; i < Superclass::m_NumberOfPixelsCounted; ++i)
    {
      for (unsigned int j = 0; j < m_G; ++j)
      {
        Amm(i, j) = A(i, j) - mean(j);
      }
    }
  }

  /** Compute covariance matrix C */
  MatrixType C(Amm.transpose() * Amm);
  C /= static_cast<RealType>(RealType(Superclass::m_NumberOfPixelsCounted) - 1.0);

  vnl_diag_matrix<RealType> S(m_G, RealType{});
  for (unsigned int j = 0; j < m_G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  /** Compute correlation matrix K */
  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType sumEigenValuesUsed{};
  for (unsigned int i = 1; i < m_NumEigenValues + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(m_G - i);
  }

  measure = m_G - sumEigenValuesUsed;

  /** Return the measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::GetDerivative(const ParametersType & parameters,
                                                    DerivativeType &       derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable. */
  MeasureType dummyvalue{};

  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(const ParametersType & parameters,
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

  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  std::vector<FixedImagePointType> SamplesOK;

  /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
  const size_t numberOfSamples{ sampleContainer->size() };
  MatrixType   datablock(numberOfSamples, m_G, vnl_matrix_null);

  /** Initialize dummy loop variables */
  unsigned int pixelIndex = 0;

  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fixedImageSample.m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
    for (unsigned int d = 0; d < m_G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[m_LastDimIndex] = d;

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
    if (numSamplesOk == m_G)
    {
      SamplesOK.push_back(fixedPoint);
      ++pixelIndex;
      Superclass::m_NumberOfPixelsCounted++;
    }

  } /** end first loop over image sample container */

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  MatrixType A(datablock.extract(Superclass::m_NumberOfPixelsCounted, m_G));

  /** Calculate standard deviation from columns */
  MatrixType Amm(Superclass::m_NumberOfPixelsCounted, m_G, vnl_matrix_null);
  {
    /** Calculate mean of from columns */
    vnl_vector<RealType> mean(m_G, RealType{});
    for (unsigned int i = 0; i < Superclass::m_NumberOfPixelsCounted; ++i)
    {
      for (unsigned int j = 0; j < m_G; ++j)
      {
        mean(j) += A(i, j);
      }
    }
    mean /= RealType(Superclass::m_NumberOfPixelsCounted);

    for (unsigned int i = 0; i < Superclass::m_NumberOfPixelsCounted; ++i)
    {
      for (unsigned int j = 0; j < m_G; ++j)
      {
        Amm(i, j) = A(i, j) - mean(j);
      }
    }
  }

  /** Compute covariance matrix C */
  MatrixType Atmm = Amm.transpose();
  MatrixType C(Atmm * Amm);
  C /= static_cast<RealType>(RealType(Superclass::m_NumberOfPixelsCounted) - 1.0);

  vnl_diag_matrix<RealType> S(m_G, RealType{});
  for (unsigned int j = 0; j < m_G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType sumEigenValuesUsed{};
  for (unsigned int i = 1; i < m_NumEigenValues + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(m_G - i);
  }

  MatrixType eigenVectorMatrix(m_G, m_NumEigenValues);
  for (unsigned int i = 1; i < m_NumEigenValues + 1; ++i)
  {
    eigenVectorMatrix.set_column(i - 1, (eig.get_eigenvector(m_G - i)).normalize());
  }

  MatrixType eigenVectorMatrixTranspose(eigenVectorMatrix.transpose());

  /** Create variables to store intermediate results in. */
  TransformJacobianType jacobian;
  DerivativeType        dMTdmu;
  DerivativeType        imageJacobian(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  std::vector<NonZeroJacobianIndicesType> nzjis(m_G, NonZeroJacobianIndicesType());

  /** Sub components of metric derivative */
  vnl_diag_matrix<DerivativeValueType> dSdmu_part1(m_G, 0.0);

  for (unsigned int d = 0; d < m_G; ++d)
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

    for (unsigned int d = 0; d < m_G; ++d)
    {
      /** Initialize some variables. */
      RealType                  movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[m_LastDimIndex] = d;

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
        for (unsigned int z = 0; z < m_NumEigenValues; ++z)
        {
          derivative[nzjis[d][p]] += vSAtmm[z][pixelIndex] * dMTdmu[p] * Sv[d][z] +
                                     vdSdmu_part1[z][d] * Atmm[d][pixelIndex] * dMTdmu[p] * CSv[d][z];
        } // end loop over eigenvalues

      } // end loop over non-zero jacobian indices

    } // end loop over last dimension

  } // end second for loop over sample container

  derivative *= -(2.0 / (DerivativeValueType(Superclass::m_NumberOfPixelsCounted) - 1.0)); // normalize
  measure = m_G - sumEigenValuesUsed;

  /** Subtract mean from derivative elements. */
  if (m_UseZeroAverageDisplacementConstraint)
  {
    if (!m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = m_GridSize[m_LastDimIndex];
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
        mean /= static_cast<RealType>(lastDimGridSize);

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
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / m_G;
      DerivativeType     mean(numParametersPerLastDimension, 0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < m_G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          mean[c % numParametersPerLastDimension] += derivative[c];
        }
      }
      mean /= static_cast<RealType>(m_G);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < m_G; ++t)
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

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(const ParametersType & parameters,
                                                            MeasureType &          value,
                                                            DerivativeType &       derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!Superclass::m_UseMultiThread)
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

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::ThreadedGetSamples(ThreadIdType threadId)
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const size_t                sampleContainerSize{ sampleContainer->size() };

  /** Get the samples for this thread. */
  const auto nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto pos_begin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto pos_end = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto threader_fbegin = beginOfSampleContainer + pos_begin;
  const auto threader_fend = beginOfSampleContainer + pos_end;

  std::vector<FixedImagePointType> SamplesOK;
  MatrixType                       datablock(nrOfSamplesPerThreads, m_G);

  unsigned int pixelIndex = 0;
  for (auto threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = threader_fiter->m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    unsigned int numSamplesOk = 0;

    /** Loop over t */
    for (unsigned int d = 0; d < m_G; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[m_LastDimIndex] = d;

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
  m_PCAMetricGetSamplesPerThreadVariables[threadId].st_NumberOfPixelsCounted = pixelIndex;
  m_PCAMetricGetSamplesPerThreadVariables[threadId].st_DataBlock = datablock.extract(pixelIndex, m_G);
  m_PCAMetricGetSamplesPerThreadVariables[threadId].st_ApprovedSamples = SamplesOK;

} // end ThreadedGetSamples()


/**
 * ******************* AfterThreadedGetSamples *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::AfterThreadedGetSamples(MeasureType & value) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  Superclass::m_NumberOfPixelsCounted = m_PCAMetricGetSamplesPerThreadVariables[0].st_NumberOfPixelsCounted;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    Superclass::m_NumberOfPixelsCounted += m_PCAMetricGetSamplesPerThreadVariables[i].st_NumberOfPixelsCounted;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples();

  MatrixType   A(Superclass::m_NumberOfPixelsCounted, m_G);
  unsigned int row_start = 0;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    A.update(m_PCAMetricGetSamplesPerThreadVariables[i].st_DataBlock, row_start, 0);
    m_PixelStartIndex[i] = row_start;
    row_start += m_PCAMetricGetSamplesPerThreadVariables[i].st_DataBlock.rows();
  }

  /** Calculate standard deviation from columns */
  MatrixType Amm(Superclass::m_NumberOfPixelsCounted, m_G, vnl_matrix_null);
  {
    /** Calculate mean of from columns */
    vnl_vector<RealType> mean(m_G, RealType{});
    for (unsigned int i = 0; i < Superclass::m_NumberOfPixelsCounted; ++i)
    {
      for (unsigned int j = 0; j < m_G; ++j)
      {
        mean(j) += A(i, j);
      }
    }
    mean /= RealType(Superclass::m_NumberOfPixelsCounted);

    for (unsigned int i = 0; i < Superclass::m_NumberOfPixelsCounted; ++i)
    {
      for (unsigned int j = 0; j < m_G; ++j)
      {
        Amm(i, j) = A(i, j) - mean(j);
      }
    }
  }

  /** Compute covariancematrix C */
  m_Atmm = Amm.transpose();
  MatrixType C(m_Atmm * Amm);
  C /= static_cast<RealType>(RealType(Superclass::m_NumberOfPixelsCounted) - 1.0);

  vnl_diag_matrix<RealType> S(m_G, RealType{});
  for (unsigned int j = 0; j < m_G; ++j)
  {
    S(j, j) = 1.0 / sqrt(C(j, j));
  }

  MatrixType K(S * C * S);

  /** Compute first eigenvalue and eigenvector of K */
  vnl_symmetric_eigensystem<RealType> eig(K);

  RealType   sumEigenValuesUsed{};
  MatrixType eigenVectorMatrix(m_G, m_NumEigenValues);
  for (unsigned int i = 1; i < m_NumEigenValues + 1; ++i)
  {
    sumEigenValuesUsed += eig.get_eigenvalue(m_G - i);
    eigenVectorMatrix.set_column(i - 1, (eig.get_eigenvector(m_G - i)).normalize());
  }

  value = m_G - sumEigenValuesUsed;

  MatrixType eigenVectorMatrixTranspose(eigenVectorMatrix.transpose());

  /** Sub components of metric derivative */
  vnl_diag_matrix<DerivativeValueType> dSdmu_part1(m_G);

  for (unsigned int d = 0; d < m_G; ++d)
  {
    double S_sqr = S(d, d) * S(d, d);
    double S_qub = S_sqr * S(d, d);
    dSdmu_part1(d, d) = -S_qub;
  }

  m_vSAtmm = eigenVectorMatrixTranspose * S * m_Atmm;
  m_CSv = C * S * eigenVectorMatrix;
  m_Sv = S * eigenVectorMatrix;
  m_vdSdmu_part1 = eigenVectorMatrixTranspose * dSdmu_part1;

} // end AfterThreadedGetSamples()


/**
 * **************** GetSamplesThreaderCallback *******
 */

template <typename TFixedImage, typename TMovingImage>
ITK_THREAD_RETURN_TYPE
PCAMetric<TFixedImage, TMovingImage>::GetSamplesThreaderCallback(void * arg)
{
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadId = infoStruct.WorkUnitID;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<PCAMetricMultiThreaderParameterType *>(infoStruct.UserData);

  userData.m_Metric->ThreadedGetSamples(threadId);

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // GetSamplesThreaderCallback()


/**
 * *********************** LaunchGetSamplesThreaderCallback***************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::LaunchGetSamplesThreaderCallback() const
{
  /** Setup local threader. */
  // \todo: is a global threader better performance-wise? check
  auto local_threader = MultiThreaderBase::New();
  local_threader->SetNumberOfWorkUnits(Self::GetNumberOfWorkUnits());
  local_threader->SetSingleMethodAndExecute(
    this->GetSamplesThreaderCallback,
    const_cast<void *>(static_cast<const void *>(&this->m_PCAMetricThreaderParameters)));

} // end LaunchGetSamplesThreaderCallback()


/**
 * ******************* ThreadedComputeDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::ThreadedComputeDerivative(ThreadIdType threadId)
{
  /** Create variables to store intermediate results in. */
  DerivativeType & derivative = m_PCAMetricGetSamplesPerThreadVariables[threadId].st_Derivative;
  derivative.Fill(0.0);

  /** Initialize some variables. */
  RealType                  movingImageValue;
  MovingImageDerivativeType movingImageDerivative;

  TransformJacobianType      jacobian;
  DerivativeType             imageJacobian(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  NonZeroJacobianIndicesType nzjis(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());

  unsigned int dummyindex = 0;
  /** Second loop over fixed image samples. */
  for (unsigned int pixelIndex = m_PixelStartIndex[threadId];
       pixelIndex <
       (m_PixelStartIndex[threadId] + m_PCAMetricGetSamplesPerThreadVariables[threadId].st_ApprovedSamples.size());
       ++pixelIndex)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = m_PCAMetricGetSamplesPerThreadVariables[threadId].st_ApprovedSamples[dummyindex];

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    for (unsigned int d = 0; d < m_G; ++d)
    {
      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[m_LastDimIndex] = d;

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
        DerivativeValueType sum = 0.0;
        for (unsigned int z = 0; z < m_NumEigenValues; ++z)
        {
          sum += m_vSAtmm[z][pixelIndex] * imageJacobian[p] * m_Sv[d][z] +
                 m_vdSdmu_part1[z][d] * m_Atmm[d][pixelIndex] * imageJacobian[p] * m_CSv[d][z];
        } // end loop over eigenvalues
        derivative[nzjis[p]] += sum;
      } // end loop over non-zero jacobian indices

    } // end loop over last dimension
    ++dummyindex;

  } // end second for loop over sample container

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedComputeDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::AfterThreadedComputeDerivative(DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  derivative = m_PCAMetricGetSamplesPerThreadVariables[0].st_Derivative;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    derivative += m_PCAMetricGetSamplesPerThreadVariables[i].st_Derivative;
  }

  derivative *= -(2.0 / (DerivativeValueType(Superclass::m_NumberOfPixelsCounted) - 1.0)); // normalize

  /** Subtract mean from derivative elements. */
  if (m_UseZeroAverageDisplacementConstraint)
  {
    if (!m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = m_GridSize[m_LastDimIndex];
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
        mean /= static_cast<RealType>(lastDimGridSize);

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
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / m_G;
      DerivativeType     mean(numParametersPerLastDimension, 0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < m_G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          mean[c % numParametersPerLastDimension] += derivative[c];
        }
      }
      mean /= static_cast<RealType>(m_G);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < m_G; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          derivative[c] -= mean[c % numParametersPerLastDimension];
        }
      }
    }
  }
} // end AftherThreadedComputeDerivative()


/**
 * **************** ComputeDerivativeThreaderCallback *******
 */

template <typename TFixedImage, typename TMovingImage>
ITK_THREAD_RETURN_TYPE
PCAMetric<TFixedImage, TMovingImage>::ComputeDerivativeThreaderCallback(void * arg)
{
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadId = infoStruct.WorkUnitID;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<PCAMetricMultiThreaderParameterType *>(infoStruct.UserData);

  userData.m_Metric->ThreadedComputeDerivative(threadId);

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end omputeDerivativeThreaderCallback()


/**
 * ************** LaunchComputeDerivativeThreaderCallback **********
 */

template <typename TFixedImage, typename TMovingImage>
void
PCAMetric<TFixedImage, TMovingImage>::LaunchComputeDerivativeThreaderCallback() const
{
  /** Setup local threader and launch. */
  // \todo: is a global threader better performance-wise? check
  auto local_threader = MultiThreaderBase::New();
  local_threader->SetNumberOfWorkUnits(Self::GetNumberOfWorkUnits());
  local_threader->SetSingleMethodAndExecute(
    this->ComputeDerivativeThreaderCallback,
    const_cast<void *>(static_cast<const void *>(&this->m_PCAMetricThreaderParameters)));

} // end LaunchComputeDerivativeThreaderCallback()


} // end namespace itk

#endif // itkPCAMetric_hxx
