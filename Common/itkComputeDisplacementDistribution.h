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
#ifndef itkComputeDisplacementDistribution_h
#define itkComputeDisplacementDistribution_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"

#include "itkImageGridSampler.h"
#include "itkImageRandomSamplerBase.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkImageFullSampler.h"
#include "itkPlatformMultiThreader.h"

#include <vector>

namespace itk
{
/**\class ComputeDisplacementDistribution
 * \brief This is a helper class for the automatic parameter estimation of the ASGD optimizer.
 *
 * More specifically this class computes the Jacobian terms related to the automatic
 * parameter estimation for the adaptive stochastic gradient descent optimizer.
 * Details can be found in the TMI paper
 *
 * [1] Y. Qiao, B. van Lew, B.P.F. Lelieveldt and M. Staring
 * "Fast Automatic Step Size Estimation for Gradient Descent Optimization of Image Registration,"
 * IEEE Transactions on Medical Imaging, vol. 35, no. 2, pp. 391 - 403, February 2016.
 * http://elastix.lumc.nl/marius/publications/2016_j_TMIa.php
 *
 */

template <class TFixedImage, class TTransform>
class ITK_TEMPLATE_EXPORT ComputeDisplacementDistribution : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ComputeDisplacementDistribution);

  /** Standard ITK.*/
  using Self = ComputeDisplacementDistribution;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ComputeDisplacementDistribution, ScaledSingleValuedNonLinearOptimizer);

  /** typedef  */
  using FixedImageType = TFixedImage;
  using FixedImagePixelType = typename FixedImageType::PixelType;
  using TransformType = TTransform;
  using TransformPointer = typename TransformType::Pointer;
  using FixedImageRegionType = typename FixedImageType::RegionType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::ScalesType;

  /** Type for the mask of the fixed image. Only pixels that are "inside"
   * this mask will be considered for the computation of the Jacobian terms.
   */
  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);
  using FixedImageMaskType = SpatialObject<Self::FixedImageDimension>;
  using FixedImageMaskPointer = typename FixedImageMaskType::Pointer;
  using FixedImageMaskConstPointer = typename FixedImageMaskType::ConstPointer;
  using NonZeroJacobianIndicesType = typename TransformType::NonZeroJacobianIndicesType;

  /** Set the fixed image. */
  itkSetConstObjectMacro(FixedImage, FixedImageType);

  /** Set the transform. */
  itkSetObjectMacro(Transform, TransformType);

  /** Set/Get the fixed image mask. */
  itkSetObjectMacro(FixedImageMask, FixedImageMaskType);
  itkSetConstObjectMacro(FixedImageMask, FixedImageMaskType);
  itkGetConstObjectMacro(FixedImageMask, FixedImageMaskType);

  /** Set some parameters. */
  itkSetMacro(NumberOfJacobianMeasurements, SizeValueType);

  /** Set the region over which the metric will be computed. */
  void
  SetFixedImageRegion(const FixedImageRegionType & region)
  {
    if (region != this->m_FixedImageRegion)
    {
      this->m_FixedImageRegion = region;
    }
  }


  /** Get the region over which the metric will be computed. */
  itkGetConstReferenceMacro(FixedImageRegion, FixedImageRegionType);

  /** The main function that performs the multi-threaded computation. */
  virtual void
  Compute(const ParametersType & mu, double & jacg, double & maxJJ, std::string method);

  /** The main function that performs the single-threaded computation. */
  virtual void
  ComputeSingleThreaded(const ParametersType & mu, double & jacg, double & maxJJ, std::string method);

  virtual void
  ComputeUsingSearchDirection(const ParametersType & mu, double & jacg, double & maxJJ, std::string methods);

  /** Set the number of threads. */
  void
  SetNumberOfWorkUnits(ThreadIdType numberOfThreads)
  {
    this->m_Threader->SetNumberOfWorkUnits(numberOfThreads);
  }


  virtual void
  BeforeThreadedCompute(const ParametersType & mu);

  virtual void
  AfterThreadedCompute(double & jacg, double & maxJJ);

protected:
  ComputeDisplacementDistribution();
  ~ComputeDisplacementDistribution() override = default;

  /** Typedefs for multi-threading. */
  using ThreaderType = itk::PlatformMultiThreader;
  using ThreadInfoType = ThreaderType::WorkUnitInfo;

  typename FixedImageType::ConstPointer   m_FixedImage;
  FixedImageRegionType                    m_FixedImageRegion;
  FixedImageMaskConstPointer              m_FixedImageMask;
  TransformPointer                        m_Transform;
  ScaledSingleValuedCostFunction::Pointer m_CostFunction;
  SizeValueType                           m_NumberOfJacobianMeasurements;
  DerivativeType                          m_ExactGradient;
  SizeValueType                           m_NumberOfParameters;
  ThreaderType::Pointer                   m_Threader;

  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImagePointType = typename FixedImageType::PointType;
  using JacobianType = typename TransformType::JacobianType;
  using JacobianValueType = typename JacobianType::ValueType;

  /** Samplers. */
  using ImageSamplerBaseType = ImageSamplerBase<FixedImageType>;
  using ImageSamplerBasePointer = typename ImageSamplerBaseType::Pointer;

  using ImageFullSamplerType = ImageFullSampler<FixedImageType>;
  using ImageFullSamplerPointer = typename ImageFullSamplerType::Pointer;

  using ImageRandomSamplerBaseType = ImageRandomSamplerBase<FixedImageType>;
  using ImageRandomSamplerBasePointer = typename ImageRandomSamplerBaseType::Pointer;

  using ImageGridSamplerType = ImageGridSampler<FixedImageType>;
  using ImageGridSamplerPointer = typename ImageGridSamplerType::Pointer;
  using ImageSampleContainerType = typename ImageGridSamplerType ::ImageSampleContainerType;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;

  /** Typedefs for support of sparse Jacobians and AdvancedTransforms. */
  using TransformJacobianType = JacobianType;
  using CoordinateRepresentationType = typename TransformType::ScalarType;
  using NumberOfParametersType = typename TransformType::NumberOfParametersType;

  /** Sample the fixed image to compute the Jacobian terms. */
  // \todo: note that this is an exact copy of itk::ComputeJacobianTerms
  // in the future it would be better to refactoring this part of the code
  virtual void
  SampleFixedImageForJacobianTerms(ImageSampleContainerPointer & sampleContainer);

  /** Launch MultiThread Compute. */
  void
  LaunchComputeThreaderCallback() const;

  /** Compute threader callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ComputeThreaderCallback(void * arg);

  /** The threaded implementation of Compute(). */
  virtual inline void
  ThreadedCompute(ThreadIdType threadID);

  /** Initialize some multi-threading related parameters. */
  virtual void
  InitializeThreadingParameters();

  /** To give the threads access to all member variables and functions. */
  struct MultiThreaderParameterType
  {
    Self * st_Self;
  };
  struct ComputePerThreadStruct
  {
    /**  Used for accumulating variables. */
    double        st_MaxJJ;
    double        st_Displacement;
    double        st_DisplacementSquared;
    SizeValueType st_NumberOfPixelsCounted;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, ComputePerThreadStruct, PaddedComputePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT, PaddedComputePerThreadStruct, AlignedComputePerThreadStruct);

private:
  mutable MultiThreaderParameterType m_ThreaderParameters;

  mutable std::vector<AlignedComputePerThreadStruct> m_ComputePerThreadVariables;

  SizeValueType               m_NumberOfPixelsCounted;
  bool                        m_UseMultiThread;
  ImageSampleContainerPointer m_SampleContainer;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkComputeDisplacementDistribution.hxx"
#endif

#endif // end #ifndef itkComputeDisplacementDistribution_h
