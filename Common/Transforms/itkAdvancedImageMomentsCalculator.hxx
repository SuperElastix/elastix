/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkAdvancedImageMomentsCalculator_hxx
#define itkAdvancedImageMomentsCalculator_hxx

#include "itkAdvancedImageMomentsCalculator.h"

#include <vnl/algo/vnl_real_eigensystem.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

//----------------------------------------------------------------------
// Construct without computing moments
template <typename TImage>
AdvancedImageMomentsCalculator<TImage>::AdvancedImageMomentsCalculator()
{
  m_Valid = false;
  m_Image = nullptr;
  m_SpatialObjectMask = nullptr;
  m_M0 = NumericTraits<ScalarType>::ZeroValue();
  m_M1.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_M2.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());
  m_Cg.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_Cm.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());
  m_Pm.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_Pa.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());

  /** Threading related variables. */
  this->m_UseMultiThread = true;
  this->m_Threader = ThreaderType::New();

  /** Initialize the m_ThreaderParameters. */
  this->m_ThreaderParameters.st_Self = this;

  // Multi-threading structs
  this->m_CenterOfGravityUsesLowerThreshold = false;
  this->m_NumberOfSamplesForCenteredTransformInitialization = 10000;
  this->m_LowerThresholdForCenterGravity = 500;
}

/**
 * ************************* InitializeThreadingParameters ************************
 */

template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::InitializeThreadingParameters()
{
  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   *
   * This function is only to be called at the start of each resolution.
   * Re-initialization of the potentially large vectors is performed after
   * each iteration, in the accumulate functions, in a multi-threaded fashion.
   * This has performance benefits for larger vector sizes.
   */
  const ThreadIdType numberOfThreads = this->m_Threader->GetNumberOfWorkUnits();

  // For each thread, assign a struct of zero-initialized values.
  m_ComputePerThreadVariables.assign(numberOfThreads, AlignedComputePerThreadStruct());

} // end InitializeThreadingParameters()

//----------------------------------------------------------------------
// Compute moments for a new or modified image
template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::ComputeSingleThreaded()
{
  if (this->m_CenterOfGravityUsesLowerThreshold)
  {
    auto thresholdFilter = BinaryThresholdImageFilterType::New();
    thresholdFilter->SetInput(this->m_Image);
    thresholdFilter->SetLowerThreshold(this->m_LowerThresholdForCenterGravity);
    thresholdFilter->SetInsideValue(1);
    thresholdFilter->SetOutsideValue(0);
    thresholdFilter->Update();
    this->SetImage(thresholdFilter->GetOutput());
  }

  m_M0 = NumericTraits<ScalarType>::ZeroValue();
  m_M1.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_M2.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());
  m_Cg.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_Cm.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());

  using IndexType = typename ImageType::IndexType;

  if (!m_Image)
  {
    return;
  }

  ImageRegionConstIteratorWithIndex<ImageType> it(m_Image, m_Image->GetRequestedRegion());

  while (!it.IsAtEnd())
  {
    double value = it.Value();

    IndexType indexPosition = it.GetIndex();

    Point<double, ImageDimension> physicalPosition;
    m_Image->TransformIndexToPhysicalPoint(indexPosition, physicalPosition);

    if (m_SpatialObjectMask.IsNull() || m_SpatialObjectMask->IsInsideInWorldSpace(physicalPosition))
    {
      m_M0 += value;

      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        m_M1[i] += static_cast<double>(indexPosition[i]) * value;
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          double weight = value * static_cast<double>(indexPosition[i]) * static_cast<double>(indexPosition[j]);
          m_M2[i][j] += weight;
        }
      }

      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        m_Cg[i] += physicalPosition[i] * value;
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          double weight = value * physicalPosition[i] * physicalPosition[j];
          m_Cm[i][j] += weight;
        }
      }
    }

    ++it;
  }
  DoPostProcessing();
}

//----------------------------------------------------------------------
// Compute moments for a new or modified image
template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::Compute()
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->ComputeSingleThreaded();
  }

  /** Initialize multi-threading. */
  this->InitializeThreadingParameters();

  /** Tackle stuff needed before multi-threading. */
  this->BeforeThreadedCompute();

  /** Launch multi-threaded computation. */
  this->LaunchComputeThreaderCallback();

  /** Gather the values from all threads. */
  this->AfterThreadedCompute();

} // end Compute()

/**
 * *********************** BeforeThreadedCompute***************
 */
template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::BeforeThreadedCompute()
{
  m_M0 = NumericTraits<ScalarType>::ZeroValue();
  m_M1.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_M2.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());
  m_Cg.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  m_Cm.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());

  if (!m_Image)
  {
    return;
  }

  if (this->m_CenterOfGravityUsesLowerThreshold)
  {
    auto thresholdFilter = BinaryThresholdImageFilterType::New();
    thresholdFilter->SetInput(this->m_Image);
    thresholdFilter->SetLowerThreshold(this->m_LowerThresholdForCenterGravity);
    thresholdFilter->SetInsideValue(1);
    thresholdFilter->SetOutsideValue(0);
    thresholdFilter->Update();
    this->SetImage(thresholdFilter->GetOutput());
  }

  this->SampleImage(this->m_SampleContainer);
} // end BeforeThreadedCompute()

/**
 * *********************** LaunchComputeThreaderCallback***************
 */

template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::LaunchComputeThreaderCallback() const
{
  /** Setup threader. */
  this->m_Threader->SetSingleMethod(this->ComputeThreaderCallback,
                                    const_cast<void *>(static_cast<const void *>(&this->m_ThreaderParameters)));

  /** Launch. */
  this->m_Threader->SingleMethodExecute();

} // end LaunchComputeThreaderCallback()

/**
 * ************ ComputeThreaderCallback ****************************
 */

template <typename TImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedImageMomentsCalculator<TImage>::ComputeThreaderCallback(void * arg)
{
  /** Get the current thread id and user data. */
  ThreadInfoType *             infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType                 threadID = infoStruct->WorkUnitID;
  MultiThreaderParameterType * temp = static_cast<MultiThreaderParameterType *>(infoStruct->UserData);

  /** Call the real implementation. */
  temp->st_Self->ThreadedCompute(threadID);

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end ComputeThreaderCallback()

/**
 * ************ ThreadedCompute ****************************
 */
template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::ThreadedCompute(ThreadIdType threadId)
{
  if (!this->m_Image)
  {
    return;
  }

  ScalarType M0 = 0;
  VectorType M1, Cg;
  M1.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  Cg.Fill(NumericTraits<typename VectorType::ValueType>::ZeroValue());
  MatrixType M2, Cm;
  M2.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());
  Cm.Fill(NumericTraits<typename MatrixType::ValueType>::ZeroValue());
  unsigned long numberOfPixelsCounted = 0;

  /** Get sample container size, number of threads, and output space dimension. */
  const SizeValueType sampleContainerSize = this->m_SampleContainer->Size();
  const ThreadIdType  numberOfThreads = this->m_Threader->GetNumberOfWorkUnits();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(numberOfThreads)));

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = this->m_SampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = this->m_SampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  for (threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    double value = threader_fiter->Value().m_ImageValue;
    // IndexType indexPosition = threader_fiter->GetIndex();
    Point<double, ImageDimension> physicalPosition = threader_fiter->Value().m_ImageCoordinates;

    if (m_SpatialObjectMask.IsNull() || m_SpatialObjectMask->IsInsideInWorldSpace(physicalPosition))
    {
      M0 += value;

      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        Cg[i] += physicalPosition[i] * value;
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          double weight = value * physicalPosition[i] * physicalPosition[j];
          Cm[i][j] += weight;
        }
      }
      ++numberOfPixelsCounted;
    }
  }
  /** Update the thread struct once. */
  AlignedComputePerThreadStruct computePerThreadStruct;
  computePerThreadStruct.st_M0 = M0;
  computePerThreadStruct.st_M1 = M1;
  computePerThreadStruct.st_M2 = M2;
  computePerThreadStruct.st_Cg = Cg;
  computePerThreadStruct.st_Cm = Cm;
  computePerThreadStruct.st_NumberOfPixelsCounted = numberOfPixelsCounted;
  m_ComputePerThreadVariables[threadId] = computePerThreadStruct;

} // end ThreadedCompute()

/**
 * *********************** AfterThreadedCompute***************
 */

template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::AfterThreadedCompute()
{
  /** Accumulate thread results. */
  for (auto & computePerThreadStruct : m_ComputePerThreadVariables)
  {
    this->m_M0 += computePerThreadStruct.st_M0;
    for (unsigned int i = 0; i < ImageDimension; ++i)
    {
      this->m_M1[i] += computePerThreadStruct.st_M1[i];
      this->m_Cg[i] += computePerThreadStruct.st_Cg[i];
      computePerThreadStruct.st_M1[i] = 0;
      computePerThreadStruct.st_Cg[i] = 0;
      for (unsigned int j = 0; j < ImageDimension; ++j)
      {
        this->m_M2[i][j] += computePerThreadStruct.st_M2[i][j];
        this->m_Cm[i][j] += computePerThreadStruct.st_Cm[i][j];
        computePerThreadStruct.st_M2[i][j] = 0;
        computePerThreadStruct.st_Cm[i][j] = 0;
      }
      computePerThreadStruct.st_M0 = 0;
    }
  }
  DoPostProcessing();
}


template <typename TImage>
void
AdvancedImageMomentsCalculator<TImage>::DoPostProcessing()
{
  // Throw an error if the total mass is zero
  if (this->m_M0 == 0.0)
  {
    itkExceptionMacro(
      << "Compute(): Total Mass of the image was zero. Aborting here to prevent division by zero later on.");
  }

  // Normalize using the total mass
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    m_Cg[i] /= m_M0;
    m_M1[i] /= m_M0;
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      m_M2[i][j] /= m_M0;
      m_Cm[i][j] /= m_M0;
    }
  }

  // Center the second order moments
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      m_M2[i][j] -= m_M1[i] * m_M1[j];
      m_Cm[i][j] -= m_Cg[i] * m_Cg[j];
    }
  }

  // Compute principal moments and axes
  vnl_symmetric_eigensystem<double> eigen(m_Cm.GetVnlMatrix().as_ref());
  vnl_diag_matrix<double>           pm = eigen.D;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    m_Pm[i] = pm(i) * m_M0;
  }
  m_Pa = eigen.V.transpose();

  // Add a final reflection if needed for a proper rotation,
  // by multiplying the last row by the determinant
  vnl_real_eigensystem                  eigenrot(m_Pa.GetVnlMatrix().as_ref());
  vnl_diag_matrix<std::complex<double>> eigenval = eigenrot.D;
  std::complex<double>                  det(1.0, 0.0);

  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    det *= eigenval(i);
  }

  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    m_Pa[ImageDimension - 1][i] *= std::real(det);
  }

  /* Remember that the moments are valid */
  m_Valid = true;
}

//---------------------------------------------------------------------
// Get sum of intensities
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetTotalMass() const -> ScalarType
{
  if (!m_Valid)
  {
    itkExceptionMacro(<< "GetTotalMass() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_M0;
}

//--------------------------------------------------------------------
// Get first moments about origin, in index coordinates
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetFirstMoments() const -> VectorType
{
  if (!m_Valid)
  {
    itkExceptionMacro(<< "GetFirstMoments() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_M1;
}

//--------------------------------------------------------------------
// Get second moments about origin, in index coordinates
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetSecondMoments() const -> MatrixType
{
  if (!m_Valid)
  {
    itkExceptionMacro(<< "GetSecondMoments() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_M2;
}

//--------------------------------------------------------------------
// Get center of gravity, in physical coordinates
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetCenterOfGravity() const -> VectorType
{
  if (!m_Valid)
  {
    itkExceptionMacro(<< "GetCenterOfGravity() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_Cg;
}

//--------------------------------------------------------------------
// Get second central moments, in physical coordinates
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetCentralMoments() const -> MatrixType
{
  if (!m_Valid)
  {
    itkExceptionMacro(<< "GetCentralMoments() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_Cm;
}

//--------------------------------------------------------------------
// Get principal moments, in physical coordinates
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetPrincipalMoments() const -> VectorType
{
  if (!m_Valid)
  {
    itkExceptionMacro(
      << "GetPrincipalMoments() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_Pm;
}

//--------------------------------------------------------------------
// Get principal axes, in physical coordinates
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetPrincipalAxes() const -> MatrixType
{
  if (!m_Valid)
  {
    itkExceptionMacro(<< "GetPrincipalAxes() invoked, but the moments have not been computed. Call Compute() first.");
  }
  return m_Pa;
}

//--------------------------------------------------------------------
// Get principal axes to physical axes transform
template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetPrincipalAxesToPhysicalAxesTransform() const -> AffineTransformPointer
{
  typename AffineTransformType::MatrixType matrix;
  typename AffineTransformType::OffsetType offset;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    offset[i] = m_Cg[i];
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      matrix[j][i] = m_Pa[i][j]; // Note the transposition
    }
  }

  AffineTransformPointer result = AffineTransformType::New();

  result->SetMatrix(matrix);
  result->SetOffset(offset);

  return result;
}

//--------------------------------------------------------------------
// Get physical axes to principal axes transform

template <typename TImage>
auto
AdvancedImageMomentsCalculator<TImage>::GetPhysicalAxesToPrincipalAxesTransform() const -> AffineTransformPointer
{
  typename AffineTransformType::MatrixType matrix;
  typename AffineTransformType::OffsetType offset;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    offset[i] = m_Cg[i];
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      matrix[j][i] = m_Pa[i][j]; // Note the transposition
    }
  }

  AffineTransformPointer result = AffineTransformType::New();
  result->SetMatrix(matrix);
  result->SetOffset(offset);

  AffineTransformPointer inverse = AffineTransformType::New();
  result->GetInverse(inverse);

  return inverse;
}

/**
 * ************************* SampleImage *********************
 */
template <typename TInputImage>
void
AdvancedImageMomentsCalculator<TInputImage>::SampleImage(ImageSampleContainerPointer & sampleContainer)
{
  /** Set up grid sampler. */
  ImageGridSamplerPointer sampler = ImageGridSamplerType::New();
  //  ImageFullSamplerPointer sampler = ImageFullSamplerType::New();
  sampler->SetInput(this->m_Image);
  sampler->SetInputImageRegion(this->m_Image->GetRequestedRegion());
  // sampler->SetMask(this->m_Image->GetSpatialObjectMask());

  /** Determine grid spacing of sampler such that the desired
   * NumberOfJacobianMeasurements is achieved approximately.
   * Note that the actually obtained number of samples may be lower, due to masks.
   * This is taken into account at the end of this function.
   */
  SizeValueType nrofsamples = this->m_NumberOfSamplesForCenteredTransformInitialization;
  sampler->SetNumberOfSamples(nrofsamples);

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();

  if (nrofsamples == 0)
  {
    itkExceptionMacro(<< "No valid voxels (0/" << this->m_NumberOfSamplesForCenteredTransformInitialization
                      << ") found to estimate the AutomaticTransformInitialization parameters.");
  }
} // end SampleImage()

template <typename TInputImage>
void
AdvancedImageMomentsCalculator<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Image: " << m_Image.GetPointer() << std::endl;
  os << indent << "Valid: " << m_Valid << std::endl;
  os << indent << "Zeroth Moment about origin: " << m_M0 << std::endl;
  os << indent << "First Moment about origin: " << m_M1 << std::endl;
  os << indent << "Second Moment about origin: " << m_M2 << std::endl;
  os << indent << "Center of Gravity: " << m_Cg << std::endl;
  os << indent << "Second central moments: " << m_Cm << std::endl;
  os << indent << "Principal Moments: " << m_Pm << std::endl;
  os << indent << "Principal axes: " << m_Pa << std::endl;
}
} // end namespace itk

#endif
