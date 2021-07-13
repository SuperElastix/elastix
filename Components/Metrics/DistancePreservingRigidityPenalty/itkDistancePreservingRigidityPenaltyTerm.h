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
  /** Standard itk stuff. */
  typedef DistancePreservingRigidityPenaltyTerm          Self;
  typedef TransformPenaltyTerm<TFixedImage, TScalarType> Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DistancePreservingRigidityPenaltyTerm, TransformPenaltyTerm);

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer           MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImagePointer            FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::GradientPixelType            GradientPixelType;
  typedef typename Superclass::GradientImageType            GradientImageType;
  typedef typename Superclass::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::DerivativeValueType          DerivativeValueType;
  typedef typename Superclass::ParametersType               ParametersType;
  typedef typename Superclass::FixedImagePixelType          FixedImagePixelType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::ScalarType                   ScalarType;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass::SpatialJacobianType           SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType            SpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType            InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** Initialize the penalty term. */
  void
  Initialize(void) override;

  /** Typedef's for B-spline transform. */
  typedef AdvancedBSplineDeformableTransform<ScalarType, FixedImageDimension, 3> BSplineTransformType;
  typedef typename BSplineTransformType::Pointer                                 BSplineTransformPointer;
  typedef typename BSplineTransformType::SpacingType                             GridSpacingType;
  typedef typename BSplineTransformType::ImageType                               CoefficientImageType;
  typedef typename CoefficientImageType::Pointer                                 CoefficientImagePointer;
  typedef typename CoefficientImageType::SpacingType                             CoefficientImageSpacingType;
  typedef AdvancedCombinationTransform<ScalarType, FixedImageDimension>          CombinationTransformType;

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
  typedef Image<signed short, itkGetStaticConstMacro(MovingImageDimension)> BSplineKnotImageType;
  typedef typename BSplineKnotImageType::Pointer                            BSplineKnotImagePointer;
  typedef typename BSplineKnotImageType::RegionType                         BSplineKnotImageRegionType;

  /** penalty grid image */
  typedef Image<signed short, itkGetStaticConstMacro(MovingImageDimension)> PenaltyGridImageType;
  typedef typename PenaltyGridImageType::Pointer                            PenaltyGridImagePointer;
  typedef typename PenaltyGridImageType::RegionType                         PenaltyGridImageRegionType;

  /** Define the segmented image. */
  typedef Image<signed short, itkGetStaticConstMacro(MovingImageDimension)> SegmentedImageType;
  typedef typename SegmentedImageType::Pointer                              SegmentedImagePointer;
  typedef typename SegmentedImageType::RegionType                           SegmentedImageRegionType;

  /** Connect the Segmented Image. */
  itkSetObjectMacro(SegmentedImage, SegmentedImageType);

  /** Get the Segmented Image. */
  itkGetModifiableObjectMacro(SegmentedImage, SegmentedImageType);

  /** Connect the Sampled Segmented Image. */
  itkSetObjectMacro(SampledSegmentedImage, SegmentedImageType);

  /** Get the Sampled Segmented Image. */
  itkGetModifiableObjectMacro(SampledSegmentedImage, SegmentedImageType);

  itkGetMacro(NumberOfRigidGrids, unsigned int);

protected:
  /** The constructor. */
  DistancePreservingRigidityPenaltyTerm();

  /** The destructor. */
  ~DistancePreservingRigidityPenaltyTerm() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** The deleted copy constructor. */
  DistancePreservingRigidityPenaltyTerm(const Self &) = delete;

  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

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
