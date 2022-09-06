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
#ifndef itkDistancePreservingRigidityPenaltyTerm_h
#define itkDistancePreservingRigidityPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"

/** Needed for the check of a B-spline transform. */
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"

/** Needed for the filtering of the B-spline coefficients. */
#include "itkNeighborhood.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkNeighborhoodIterator.h"

#include "itkImageRegionIterator.h"
#include "itkMultiResolutionPyramidImageFilter.h"

namespace itk
{
/**
 * \class DistancePreservingRigidityPenalty
 * \brief A penalty term designed to preserve inter-voxel distances within rigid body regions.
 *
 * For more information check the paper:\n
 *  J. Kim, M. M. Matuszak, K. Saitou, and J. Balter,
 *  "Distance-preserving rigidity penalty on deformable image registration of multiple skeletal components in the neck"
 *  Medical Physics, vol. 40, no. 12, pp. 121907-1 - 121907-10, December 2013.
 * - view online: http://www.ncbi.nlm.nih.gov/pubmed/24320518
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "DistancePreservingRigidityPenalty")</tt>
 *
 * \parameter SegmentedImageName: The file name of the image to
 *    specify the rigidity index of voxels in the fixed image. The
 *  image has only non-integer values as follows:
 *    1) background: 0,
 *    2) rigid region1: 1,
 *    3) rigid region2: 2, and so on.
 *    - example: <tt>(SegmentedImageName "BoneSegmentation.mhd")</tt> \n
 *
 * \parameter PenaltyGridSpacingInVoxels: defines the grid spacing
 *  with which the rigidity penalty is calculated. In this current
 *  version, the grid spacing is set to be constant over different
 *  resolutions.
 *  - In the publication above, the grid spacing was set as [4, 4, 1].
 *
 * \author Jihun Kim, University of Michigan, Ann Arbor
 * \author Martha M. Matuszak, University of Michigan, Ann Arbor
 * \author Kazuhiro Saitou, University of Michigan, Ann Arbor
 * \author James Balter, University of Michigan, Ann Arbor
 *
 * \ingroup Metrics
 *
 */

template <class TFixedImage, class TScalarType>
class ITK_TEMPLATE_EXPORT DistancePreservingRigidityPenaltyTerm : public TransformPenaltyTerm<TFixedImage, TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DistancePreservingRigidityPenaltyTerm);

  /** Standard itk stuff. */
  using Self = DistancePreservingRigidityPenaltyTerm;
  using Superclass = TransformPenaltyTerm<TFixedImage, TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DistancePreservingRigidityPenaltyTerm, TransformPenaltyTerm);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImagePointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImagePointer;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::ScalarType;

  /** Typedefs from the AdvancedTransform. */
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** Initialize the penalty term. */
  void
  Initialize() override;

  /** Typedef's for B-spline transform. */
  using BSplineTransformType = AdvancedBSplineDeformableTransform<ScalarType, FixedImageDimension, 3>;
  using BSplineTransformPointer = typename BSplineTransformType::Pointer;
  using GridSpacingType = typename BSplineTransformType::SpacingType;
  using CoefficientImageType = typename BSplineTransformType::ImageType;
  using CoefficientImagePointer = typename CoefficientImageType::Pointer;
  using CoefficientImageSpacingType = typename CoefficientImageType::SpacingType;
  using CombinationTransformType = AdvancedCombinationTransform<ScalarType, FixedImageDimension>;

  /** The GetValue()-method returns the rigid penalty value. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** The GetDerivative()-method returns the rigid penalty derivative. */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** The GetValueAndDerivative()-method returns the rigid penalty value and its derivative. */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** Set the B-spline transform in this class.
   * This class expects a BSplineTransform! It is not suited for others.
   */
  itkSetObjectMacro(BSplineTransform, BSplineTransformType);

  /** B-spline knot image */
  using BSplineKnotImageType = Image<signed short, Self::MovingImageDimension>;
  using BSplineKnotImagePointer = typename BSplineKnotImageType::Pointer;
  using BSplineKnotImageRegionType = typename BSplineKnotImageType::RegionType;

  /** penalty grid image */
  using PenaltyGridImageType = Image<signed short, Self::MovingImageDimension>;
  using PenaltyGridImagePointer = typename PenaltyGridImageType::Pointer;
  using PenaltyGridImageRegionType = typename PenaltyGridImageType::RegionType;

  /** Define the segmented image. */
  using SegmentedImageType = Image<signed short, Self::MovingImageDimension>;
  using SegmentedImagePointer = typename SegmentedImageType::Pointer;
  using SegmentedImageRegionType = typename SegmentedImageType::RegionType;

  /** Connect the Segmented Image. */
  itkSetObjectMacro(SegmentedImage, SegmentedImageType);

  /** Get the Segmented Image. */
  itkGetModifiableObjectMacro(SegmentedImage, SegmentedImageType);

  /** Connect the Sampled Segmented Image. */
  itkSetObjectMacro(SampledSegmentedImage, SegmentedImageType);

  /** Get the Sampled Segmented Image. */
  itkGetModifiableObjectMacro(SampledSegmentedImage, SegmentedImageType);

  itkGetConstMacro(NumberOfRigidGrids, unsigned int);

protected:
  /** The constructor. */
  DistancePreservingRigidityPenaltyTerm();

  /** The destructor. */
  ~DistancePreservingRigidityPenaltyTerm() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Member variables. */
  BSplineTransformPointer m_BSplineTransform;

  mutable MeasureType m_RigidityPenaltyTermValue;

  BSplineKnotImagePointer m_BSplineKnotImage;
  PenaltyGridImagePointer m_PenaltyGridImage;
  SegmentedImagePointer   m_SegmentedImage;
  SegmentedImagePointer   m_SampledSegmentedImage;

  unsigned int m_NumberOfRigidGrids;
};

// end class DistancePreservingRigidityPenaltyTerm

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkDistancePreservingRigidityPenaltyTerm.hxx"
#endif

#endif // #ifndef itkDistancePreservingRigidityPenaltyTerm_h
