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
#ifndef itkComputeJacobianTerms_h
#define itkComputeJacobianTerms_h

#include "itkImageGridSampler.h"
#include "itkImageRandomSamplerBase.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkScaledSingleValuedNonLinearOptimizer.h"

namespace itk
{
/**\class ComputeJacobianTerms
 * \brief This is a helper class for the automatic parameter estimation of the ASGD optimizer.
 *
 * More specifically this class computes the Jacobian terms related to the automatic
 * parameter estimation for the adaptive stochastic gradient descent optimizer.
 * Details can be found in the paper.
 */

template <class TFixedImage, class TTransform>
class ITK_TEMPLATE_EXPORT ComputeJacobianTerms : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ComputeJacobianTerms);

  /** Standard ITK.*/
  using Self = ComputeJacobianTerms;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ComputeJacobianTerms, Object);

  /** typedef  */
  using FixedImageType = TFixedImage;
  using TransformType = TTransform;
  using TransformPointer = typename TransformType::Pointer;
  using FixedImageRegionType = typename FixedImageType::RegionType;

  /** Type for the mask of the fixed image. Only pixels that are "inside"
   * this mask will be considered for the computation of the Jacobian terms.
   */
  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);
  using FixedImageMaskType = SpatialObject<Self::FixedImageDimension>;
  using FixedImageMaskPointer = typename FixedImageMaskType::Pointer;
  using FixedImageMaskConstPointer = typename FixedImageMaskType::ConstPointer;

  using ScaledSingleValuedNonLinearOptimizerType = ScaledSingleValuedNonLinearOptimizer;
  using ScaledCostFunctionPointer = typename ScaledSingleValuedNonLinearOptimizerType ::ScaledCostFunctionPointer;
  using ScalesType = typename ScaledSingleValuedNonLinearOptimizerType::ScalesType;
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
  itkSetMacro(Scales, ScalesType);
  itkSetMacro(UseScales, bool);
  itkSetMacro(MaxBandCovSize, unsigned int);
  itkSetMacro(NumberOfBandStructureSamples, unsigned int);
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

  /** The main functions that performs the computation. */
  virtual void
  Compute(double & TrC, double & TrCC, double & maxJJ, double & maxJCJ);

protected:
  ComputeJacobianTerms() = default;
  ~ComputeJacobianTerms() override = default;

  typename FixedImageType::ConstPointer m_FixedImage{ nullptr };
  FixedImageRegionType                  m_FixedImageRegion;
  FixedImageMaskConstPointer            m_FixedImageMask{ nullptr };
  TransformPointer                      m_Transform{ nullptr };
  ScalesType                            m_Scales;
  bool                                  m_UseScales{ false };

  unsigned int  m_MaxBandCovSize{ 0 };
  unsigned int  m_NumberOfBandStructureSamples{ 0 };
  SizeValueType m_NumberOfJacobianMeasurements{ 0 };

  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImagePointType = typename FixedImageType::PointType;
  using JacobianType = typename TransformType::JacobianType;
  using JacobianValueType = typename JacobianType::ValueType;

  /** Samplers. */
  using ImageSamplerBaseType = ImageSamplerBase<FixedImageType>;
  using ImageSamplerBasePointer = typename ImageSamplerBaseType::Pointer;
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
  // \todo: note that this is an exact copy of itk::ComputeDisplacementDistribution
  // in the future it would be better to refactoring this part of the code.
  virtual void
  SampleFixedImageForJacobianTerms(ImageSampleContainerPointer & sampleContainer);
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkComputeJacobianTerms.hxx"
#endif

#endif // end #ifndef itkComputeJacobianTerms_h
