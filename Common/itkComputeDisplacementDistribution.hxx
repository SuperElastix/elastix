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
#ifndef itkComputeDisplacementDistribution_hxx
#define itkComputeDisplacementDistribution_hxx

#include "itkComputeDisplacementDistribution.h"

#include <string>
#include <vnl/vnl_math.h>
#include <vnl/vnl_fastops.h>
#include <vnl/vnl_diag_matrix.h>

#include "itkImageScanlineIterator.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkCropImageFilter.h"
#include "itkMirrorPadImageFilter.h"
#include "itkZeroFluxNeumannPadImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include <cassert>

namespace itk
{

/**
 * ************************* Constructor ************************
 */

template <typename TFixedImage, typename TTransform>
ComputeDisplacementDistribution<TFixedImage, TTransform>::ComputeDisplacementDistribution()
{
  /** Initialize the m_ThreaderParameters. */
  m_ThreaderParameters.st_Self = this;

} // end Constructor


/**
 * ************************* InitializeThreadingParameters ************************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::InitializeThreadingParameters()
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
  const ThreadIdType numberOfThreads = m_Threader->GetNumberOfWorkUnits();

  // For each thread, assign a struct of zero-initialized values.
  m_ComputePerThreadVariables.assign(numberOfThreads, AlignedComputePerThreadStruct());

} // end InitializeThreadingParameters()


/**
 * ************************* ComputeSingleThreaded ************************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::ComputeSingleThreaded(const ParametersType & mu,
                                                                                double &               jacg,
                                                                                double &               maxJJ,
                                                                                std::string            methods)
{
  /** This function computes four terms needed for the automatic parameter
   * estimation using voxel displacement distribution estimation method.
   * The equation number refers to the SPIE paper.
   * Term 1: jacg = mean( J_j * g ) + var( J_j * g ).
   */

  /** Initialize. */
  maxJJ = jacg = 0.0;

  /** Get samples. */
  const std::vector<ImageSampleType> samples = this->SampleFixedImageForJacobianTerms();
  const SizeValueType                nrofsamples = samples.size();

  /** Get the number of parameters. */
  const auto numberOfParameters = static_cast<unsigned int>(m_Transform->GetNumberOfParameters());

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  Superclass::m_ScaledCostFunction->SetScales(scales);

  /** Get the exact gradient. */
  m_ExactGradient.set_size(numberOfParameters);
  m_ExactGradient.Fill(0.0);
  this->GetScaledDerivative(mu, m_ExactGradient);

  static constexpr unsigned int outdim{ TTransform::OutputSpaceDimension };

  unsigned int samplenr = 0;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType        sizejacind = m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType               jacj(outdim, sizejacind, 0.0);
  NonZeroJacobianIndicesType jacind(sizejacind);
  assert((sizejacind > 0) && (jacind.front() == 0));

  /**
   * Compute maxJJ and jac*gradient
   */
  DerivativeType Jgg(outdim, 0.0);

  std::vector<double> JGG_k;
  JGG_k.reserve(nrofsamples);

  double       globalDeformation = 0.0;
  const double sqrt2 = std::sqrt(static_cast<double>(2.0));
  JacobianType jacjjacj(outdim, outdim);

  samplenr = 0;
  for (const auto & sample : samples)
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = sample.m_ImageCoordinates;
    m_Transform->GetJacobian(point, jacj, jacind);

    /** Apply scales, if necessary. */
    if (this->GetUseScales())
    {
      for (unsigned int pi = 0; pi < sizejacind; ++pi)
      {
        const unsigned int p = jacind[pi];
        jacj.scale_column(pi, 1.0 / scales[p]);
      }
    }

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math::sqr(jacj.frobenius_norm());

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt(jacjjacj, jacj, jacj);
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = std::max(maxJJ, JJ_j);

    /** Compute the matrix of jac*gradient */
    for (unsigned int i = 0; i < outdim; ++i)
    {
      double temp = 0.0;
      for (unsigned int j = 0; j < sizejacind; ++j)
      {
        int pj = jacind[j];
        temp += jacj(i, j) * m_ExactGradient(pj);
      }
      Jgg(i) = temp;
    }

    globalDeformation += Jgg.magnitude();
    JGG_k.push_back(Jgg.magnitude());
    ++samplenr;

  } // end loop over sample container

  if (methods == "95percentile")
  {
    /** Compute the 95% percentile of the distribution of JGG_k */
    auto d = static_cast<unsigned int>(nrofsamples * 0.95);
    std::sort(JGG_k.begin(), JGG_k.end());
    jacg = (JGG_k[d - 1] + JGG_k[d] + JGG_k[d + 1]) / 3.0;
  }
  else if (methods == "2sigma")
  {
    /** Compute the sigma of the distribution of JGG_k. */
    double sigma = 0.0;
    double mean_JGG = globalDeformation / samplenr;
    for (unsigned int i = 0; i < nrofsamples; ++i)
    {
      sigma += vnl_math::sqr(JGG_k[i] - mean_JGG);
    }
    sigma /= (nrofsamples - 1); // unbiased estimation
    jacg = mean_JGG + 2.0 * std::sqrt(sigma);
  }

} // end ComputeSingleThreaded()


/**
 * ************************* Compute ************************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::Compute(const ParametersType & mu,
                                                                  double &               jacg,
                                                                  double &               maxJJ,
                                                                  std::string            methods)
{
  /** Option for now to still use the single threaded code. */
  if (!m_UseMultiThread)
  {
    return this->ComputeSingleThreaded(mu, jacg, maxJJ, methods);
  }
  // The multi-threaded route only supports methods == 2sigma for now

  /** Initialize multi-threading. */
  this->InitializeThreadingParameters();

  /** Tackle stuff needed before multi-threading. */
  this->BeforeThreadedCompute(mu);

  /** Launch multi-threaded computation. */
  this->LaunchComputeThreaderCallback();

  /** Gather the jacg, maxJJ values from all threads. */
  this->AfterThreadedCompute(jacg, maxJJ);

} // end Compute()


/**
 * *********************** BeforeThreadedCompute***************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::BeforeThreadedCompute(const ParametersType & mu)
{
  /** Get the number of parameters. */
  m_NumberOfParameters =
    static_cast<unsigned int>(m_Transform->GetNumberOfParameters()); // why is this parameter needed?

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  Superclass::m_ScaledCostFunction->SetScales(scales);

  /** Get the exact gradient. */
  m_ExactGradient.set_size(m_NumberOfParameters);
  m_ExactGradient.Fill(0.0);
  this->GetScaledDerivative(mu, m_ExactGradient);

  /** Get samples. */
  m_Samples = this->SampleFixedImageForJacobianTerms();

} // end BeforeThreadedCompute()


/**
 * *********************** LaunchComputeThreaderCallback***************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::LaunchComputeThreaderCallback() const
{
  /** Setup threader and launch. */
  m_Threader->SetSingleMethodAndExecute(this->ComputeThreaderCallback, &m_ThreaderParameters);

} // end LaunchComputeThreaderCallback()


/**
 * ************ ComputeThreaderCallback ****************************
 */

template <typename TFixedImage, typename TTransform>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ComputeDisplacementDistribution<TFixedImage, TTransform>::ComputeThreaderCallback(void * arg)
{
  /** Get the current thread id and user data. */
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadID = infoStruct.WorkUnitID;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

  /** Call the real implementation. */
  userData.st_Self->ThreadedCompute(threadID);

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end ComputeThreaderCallback()


/**
 * ************************* ThreadedCompute ************************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::ThreadedCompute(ThreadIdType threadId)
{
  /** Get sample container size, number of threads, and output space dimension. */
  const SizeValueType           sampleContainerSize = m_Samples.size();
  const ThreadIdType            numberOfThreads = m_Threader->GetNumberOfWorkUnits();
  static constexpr unsigned int outdim{ TTransform::OutputSpaceDimension };

  /** Get a handle to the scales vector */
  const ScalesType & scales = this->GetScales();

  /** Get the samples for this thread. */
  const auto nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(numberOfThreads)));

  const auto pos_begin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto pos_end = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType        sizejacind = m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType               jacj(outdim, sizejacind, 0.0);
  NonZeroJacobianIndicesType jacind(sizejacind);
  assert((sizejacind > 0) && (jacind.front() == 0));

  /** Temporaries. */
  // std::vector< double > JGG_k; not here so only mean + 2 sigma is supported
  DerivativeType Jgg(outdim, 0.0);
  const double   sqrt2 = std::sqrt(static_cast<double>(2.0));
  JacobianType   jacjjacj(outdim, outdim);
  double         maxJJ = 0.0;
  double         jggMagnitude = 0.0;
  double         displacement = 0.0;
  double         displacementSquared = 0.0;
  unsigned long  numberOfPixelsCounted = 0;

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = m_Samples.cbegin();
  const auto threader_fbegin = beginOfSampleContainer + pos_begin;
  const auto threader_fend = beginOfSampleContainer + pos_end;

  /** Loop over the fixed image to calculate the mean squares. */
  for (auto threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = threader_fiter->m_ImageCoordinates;
    m_Transform->GetJacobian(point, jacj, jacind);

    /** Apply scales, if necessary. */
    if (this->GetUseScales())
    {
      for (unsigned int pi = 0; pi < sizejacind; ++pi)
      {
        const unsigned int p = jacind[pi];
        jacj.scale_column(pi, 1.0 / scales[p]);
      }
    }

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math::sqr(jacj.frobenius_norm());

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt(jacjjacj, jacj, jacj); // is this thread-safe?
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = std::max(maxJJ, JJ_j);

    /** Compute the displacement  jac * gradient. */
    for (unsigned int i = 0; i < outdim; ++i)
    {
      double temp = 0.0;
      for (unsigned int j = 0; j < sizejacind; ++j)
      {
        int pj = jacind[j];
        temp += jacj(i, j) * m_ExactGradient(pj);
      }
      Jgg(i) = temp;
    }

    /** Sum the Jgg displacement for later use. */
    jggMagnitude = Jgg.magnitude();
    displacement += jggMagnitude;
    displacementSquared += vnl_math::sqr(jggMagnitude);
    ++numberOfPixelsCounted;
  }

  /** Update the thread struct once. */
  AlignedComputePerThreadStruct computePerThreadStruct;
  computePerThreadStruct.st_MaxJJ = maxJJ;
  computePerThreadStruct.st_Displacement = displacement;
  computePerThreadStruct.st_DisplacementSquared = displacementSquared;
  computePerThreadStruct.st_NumberOfPixelsCounted = numberOfPixelsCounted;
  m_ComputePerThreadVariables[threadId] = computePerThreadStruct;
} // end ThreadedCompute()


/**
 * *********************** AfterThreadedCompute***************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::AfterThreadedCompute(double & jacg, double & maxJJ)
{
  /** Reset all variables. */
  maxJJ = 0.0;
  double displacement = 0.0;
  double displacementSquared = 0.0;
  m_NumberOfPixelsCounted = 0.0;

  /** Accumulate thread results. */
  for (const auto & computePerThreadStruct : m_ComputePerThreadVariables)
  {
    maxJJ = std::max(maxJJ, computePerThreadStruct.st_MaxJJ);
    displacement += computePerThreadStruct.st_Displacement;
    displacementSquared += computePerThreadStruct.st_DisplacementSquared;
    m_NumberOfPixelsCounted += computePerThreadStruct.st_NumberOfPixelsCounted;
  }
  // Reset all variables for the next resolution.
  std::fill_n(m_ComputePerThreadVariables.begin(), m_ComputePerThreadVariables.size(), AlignedComputePerThreadStruct());

  /** Compute the sigma of the distribution of the displacements. */
  const double meanDisplacement = displacement / m_NumberOfPixelsCounted;
  const double sigma = displacementSquared / m_NumberOfPixelsCounted - vnl_math::sqr(meanDisplacement);

  jacg = meanDisplacement + 2.0 * std::sqrt(sigma);

} // end AfterThreadedCompute()


/**
 * ************************* ComputeUsingSearchDirection ************************
 */

template <typename TFixedImage, typename TTransform>
void
ComputeDisplacementDistribution<TFixedImage, TTransform>::ComputeUsingSearchDirection(const ParametersType & mu,
                                                                                      double &               jacg,
                                                                                      double &               maxJJ,
                                                                                      std::string            methods)
{
  /** This function computes four terms needed for the automatic parameter
   * estimation using voxel displacement distribution estimation method.
   * The equation number refers to the SPIE paper.
   * Term 1: jacg = mean( J_j * g ) + var( J_j * g ).
   */

  /** Initialize. */
  maxJJ = jacg = 0.0;

  /** Get samples. */
  const std::vector<ImageSampleType> samples = this->SampleFixedImageForJacobianTerms();
  const SizeValueType                nrofsamples = samples.size();

  /** Get the number of parameters. */
  const auto numberOfParameters = static_cast<unsigned int>(m_Transform->GetNumberOfParameters());

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  Superclass::m_ScaledCostFunction->SetScales(scales);

  /** Get the exact gradient. */
  DerivativeType exactgradient(numberOfParameters);
  exactgradient = mu;

  static constexpr unsigned int outdim{ TTransform::OutputSpaceDimension };

  unsigned int samplenr = 0;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType        sizejacind = m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType               jacj(outdim, sizejacind, 0.0);
  NonZeroJacobianIndicesType jacind(sizejacind);
  assert((sizejacind > 0) && (jacind.front() == 0));

  /**
   * Compute maxJJ and jac*gradient
   */
  DerivativeType Jgg(outdim, 0.0);

  std::vector<double> JGG_k;
  JGG_k.reserve(nrofsamples);

  double globalDeformation = 0.0;

  samplenr = 0;
  for (const auto & sample : samples)
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = sample.m_ImageCoordinates;
    m_Transform->GetJacobian(point, jacj, jacind);

    /** Apply scales, if necessary. */
    if (this->GetUseScales())
    {
      for (unsigned int pi = 0; pi < sizejacind; ++pi)
      {
        const unsigned int p = jacind[pi];
        jacj.scale_column(pi, 1.0 / scales[p]);
      }
    }

    /** Compute the matrix of jac*gradient */
    for (unsigned int i = 0; i < outdim; ++i)
    {
      double temp = 0.0;
      for (unsigned int j = 0; j < sizejacind; ++j)
      {
        int pj = jacind[j];
        temp += jacj(i, j) * exactgradient(pj);
      }
      Jgg(i) = temp;
    }

    globalDeformation += Jgg.magnitude();
    JGG_k.push_back(Jgg.magnitude());
    ++samplenr;

  } // end loop over sample container

  if (methods == "95percentile")
  {
    /** Compute the 95% percentile of the distribution of JGG_k */
    auto d = static_cast<unsigned int>(nrofsamples * 0.95);
    std::sort(JGG_k.begin(), JGG_k.end());
    jacg = (JGG_k[d - 1] + JGG_k[d] + JGG_k[d + 1]) / 3.0;
  }
  else if (methods == "2sigma")
  {
    /** Compute the sigma of the distribution of JGG_k. */
    double sigma = 0.0;
    double mean_JGG = globalDeformation / samplenr;
    for (unsigned int i = 0; i < nrofsamples; ++i)
    {
      sigma += vnl_math::sqr(JGG_k[i] - mean_JGG);
    }
    sigma /= (nrofsamples - 1); // unbiased estimation
    jacg = mean_JGG + 2.0 * std::sqrt(sigma);
  }
} // end ComputeUsingSearchDirection()


/**
 * ************************* SampleFixedImageForJacobianTerms ************************
 */

template <typename TFixedImage, typename TTransform>
auto
ComputeDisplacementDistribution<TFixedImage, TTransform>::SampleFixedImageForJacobianTerms() const
  -> std::vector<ImageSampleType>
{
  /** Set up grid sampler. */
  const auto sampler = ImageGridSampler<TFixedImage>::New();
  sampler->SetInput(m_FixedImage);
  sampler->SetInputImageRegion(m_FixedImageRegion);
  sampler->SetMask(m_FixedImageMask);

  /** Determine grid spacing of sampler such that the desired
   * NumberOfJacobianMeasurements is achieved approximately.
   * Note that the actually obtained number of samples may be lower, due to masks.
   * This is taken into account at the end of this function.
   */
  sampler->SetNumberOfSamples(m_NumberOfJacobianMeasurements);

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  std::vector<ImageSampleType> & samples = Deref(sampler->GetOutput()).CastToSTLContainer();

  if (samples.empty())
  {
    itkExceptionMacro("No valid voxels (0/" << m_NumberOfJacobianMeasurements
                                            << ") found to estimate the AdaptiveStochasticGradientDescent parameters.");
  }
  return std::move(samples);

} // end SampleFixedImageForJacobianTerms()


} // end namespace itk

#endif // end #ifndef itkComputeDisplacementDistribution_hxx
