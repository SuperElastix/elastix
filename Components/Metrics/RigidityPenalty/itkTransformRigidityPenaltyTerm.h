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
#ifndef itkTransformRigidityPenaltyTerm_h
#define itkTransformRigidityPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"

/** Needed for the check of a B-spline transform. */
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"

/** Needed for the filtering of the B-spline coefficients. */
#include "itkNeighborhood.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkNeighborhoodIterator.h"

/** Include stuff needed for the construction of the rigidity coefficient image. */
#include "itkGrayscaleDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageRegionIterator.h"

namespace itk
{
/**
 * \class TransformRigidityPenaltyTerm
 * \brief A cost function that calculates a rigidity penalty term.
 *
 * A cost function that calculates a rigidity penalty term based
 * on the B-spline coefficients of a B-spline transformation.
 * This penalty term is a function of the 1st and 2nd order spatial
 * derivatives of a transformation.
 *
 * The intended use for this metric is to filter a B-spline coefficient
 * image in order to calculate a rigidity penalty term on a B-spline transform.
 *
 * The RigidityPenaltyTermValueImageFilter at each pixel location is computed by
 * convolution with some separable 1D kernels.
 *
 * The rigid penalty term penalizes deviations from a rigid
 * transformation at regions specified by the so-called rigidity images.
 *
 * This metric only works with B-splines as a transformation model.
 *
 * References:\n
 * [1] M. Staring, S. Klein and J.P.W. Pluim,
 *    "A Rigidity Penalty Term for Nonrigid Registration,"
 *    Medical Physics, vol. 34, no. 11, pp. 4098 - 4108, November 2007.
 *
 * \sa BSplineTransform
 *
 * \ingroup Metrics
 */

template <class TFixedImage, class TScalarType>
class ITK_TEMPLATE_EXPORT TransformRigidityPenaltyTerm : public TransformPenaltyTerm<TFixedImage, TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformRigidityPenaltyTerm);

  /** Standard itk stuff. */
  using Self = TransformRigidityPenaltyTerm;
  using Superclass = TransformPenaltyTerm<TFixedImage, TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformRigidityPenaltyTerm, TransformPenaltyTerm);

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

  /** Typedef's for the B-spline transform. */
  using typename Superclass::CombinationTransformType;
  using typename Superclass::BSplineOrder1TransformType;
  using typename Superclass::BSplineOrder1TransformPointer;
  using typename Superclass::BSplineOrder2TransformType;
  using typename Superclass::BSplineOrder2TransformPointer;
  using typename Superclass::BSplineOrder3TransformType;
  using typename Superclass::BSplineOrder3TransformPointer;

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
  using BSplineTransformType = BSplineOrder3TransformType;
  using BSplineTransformPointer = typename BSplineTransformType::Pointer;
  using GridSpacingType = typename BSplineTransformType::SpacingType;
  using CoefficientImageType = typename BSplineTransformType::ImageType;
  using CoefficientImagePointer = typename CoefficientImageType::Pointer;
  using CoefficientImageSpacingType = typename CoefficientImageType::SpacingType;

  /** Typedef support for neighborhoods, filters, etc. */
  using NeighborhoodType = Neighborhood<ScalarType, Self::FixedImageDimension>;
  using NeighborhoodSizeType = typename NeighborhoodType::SizeType;
  using CoefficientImageIteratorType = ImageRegionIterator<CoefficientImageType>;
  using NOIFType = NeighborhoodOperatorImageFilter<CoefficientImageType, CoefficientImageType>;
  using NeighborhoodIteratorType = NeighborhoodIterator<CoefficientImageType>;
  using RadiusType = typename NeighborhoodIteratorType::RadiusType;

  /** Typedef's for the construction of the rigidity image. */
  using RigidityImageType = CoefficientImageType;
  using RigidityImagePointer = typename RigidityImageType::Pointer;
  using RigidityPixelType = typename RigidityImageType::PixelType;
  using RigidityImageRegionType = typename RigidityImageType::RegionType;
  using RigidityImageIndexType = typename RigidityImageType::IndexType;
  using RigidityImagePointType = typename RigidityImageType::PointType;
  using RigidityImageIteratorType = ImageRegionIterator<RigidityImageType>;
  using StructuringElementType = BinaryBallStructuringElement<RigidityPixelType, Self::FixedImageDimension>;
  using SERadiusType = typename StructuringElementType::RadiusType;
  using DilateFilterType = GrayscaleDilateImageFilter<RigidityImageType, RigidityImageType, StructuringElementType>;
  using DilateFilterPointer = typename DilateFilterType::Pointer;

  /** Check stuff. */
  void
  CheckUseAndCalculationBooleans();

  /** The GetValue()-method returns the rigid penalty value. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** The GetDerivative()-method returns the rigid penalty derivative. */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** Contains calls from GetValueAndDerivative that are thread-unsafe. */
  void
  BeforeThreadedGetValueAndDerivative(const TransformParametersType & parameters) const override;

  /** The GetValueAndDerivative()-method returns the rigid penalty value and its derivative. */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** Set the B-spline transform in this class.
   * This class expects a BSplineTransform! It is not suited for others.
   */
  itkSetObjectMacro(BSplineTransform, BSplineTransformType);

  /** Set the RigidityImage in this class. */
  // itkSetObjectMacro( RigidityCoefficientImage, RigidityImageType );

  /** Set/Get the weight of the linearity condition part. */
  itkSetClampMacro(LinearityConditionWeight, ScalarType, 0.0, NumericTraits<ScalarType>::max());
  itkGetConstMacro(LinearityConditionWeight, ScalarType);

  /** Set/Get the weight of the orthonormality condition part. */
  itkSetClampMacro(OrthonormalityConditionWeight, ScalarType, 0.0, NumericTraits<ScalarType>::max());
  itkGetConstMacro(OrthonormalityConditionWeight, ScalarType);

  /** Set/Get the weight of the properness condition part. */
  itkSetClampMacro(PropernessConditionWeight, ScalarType, 0.0, NumericTraits<ScalarType>::max());
  itkGetConstMacro(PropernessConditionWeight, ScalarType);

  /** Set the usage of the linearity condition part. */
  itkSetMacro(UseLinearityCondition, bool);

  /** Set the usage of the orthonormality condition part. */
  itkSetMacro(UseOrthonormalityCondition, bool);

  /** Set the usage of the properness condition part. */
  itkSetMacro(UsePropernessCondition, bool);

  /** Set the calculation of the linearity condition part,
   * even if we don't use it.
   */
  itkSetMacro(CalculateLinearityCondition, bool);

  /** Set the calculation of the orthonormality condition part,
   * even if we don't use it.
   */
  itkSetMacro(CalculateOrthonormalityCondition, bool);

  /** Set the calculation of the properness condition part.,
   * even if we don't use it.
   */
  itkSetMacro(CalculatePropernessCondition, bool);

  /** Get the value of the linearity condition. */
  itkGetConstReferenceMacro(LinearityConditionValue, MeasureType);

  /** Get the value of the orthonormality condition. */
  itkGetConstReferenceMacro(OrthonormalityConditionValue, MeasureType);

  /** Get the value of the properness condition. */
  itkGetConstReferenceMacro(PropernessConditionValue, MeasureType);

  /** Get the gradient magnitude of the linearity condition. */
  itkGetConstReferenceMacro(LinearityConditionGradientMagnitude, MeasureType);

  /** Get the gradient magnitude of the orthonormality condition. */
  itkGetConstReferenceMacro(OrthonormalityConditionGradientMagnitude, MeasureType);

  /** Get the gradient magnitude of the properness condition. */
  itkGetConstReferenceMacro(PropernessConditionGradientMagnitude, MeasureType);

  /** Get the value of the total rigidity penalty term. */
  // itkGetConstReferenceMacro( RigidityPenaltyTermValue, MeasureType );

  /** Set if the RigidityImage's are dilated. */
  itkSetMacro(DilateRigidityImages, bool);

  /** Set the DilationRadiusMultiplier. */
  itkSetClampMacro(DilationRadiusMultiplier,
                   CoordinateRepresentationType,
                   0.1,
                   NumericTraits<CoordinateRepresentationType>::max());

  /** Set the fixed coefficient image. */
  itkSetObjectMacro(FixedRigidityImage, RigidityImageType);

  /** Set the moving coefficient image. */
  itkSetObjectMacro(MovingRigidityImage, RigidityImageType);

  /** Set to use the FixedRigidityImage or not. */
  itkSetMacro(UseFixedRigidityImage, bool);

  /** Set to use the MovingRigidityImage or not. */
  itkSetMacro(UseMovingRigidityImage, bool);

  /** Function to fill the RigidityCoefficientImage every iteration. */
  void
  FillRigidityCoefficientImage(const ParametersType & parameters) const;

protected:
  /** The constructor. */
  TransformRigidityPenaltyTerm();
  /** The destructor. */
  ~TransformRigidityPenaltyTerm() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Internal function to dilate the rigidity images. */
  virtual void
  DilateRigidityImages();

  /** Private function used for the filtering. It creates 1D separable operators F. */
  void
  Create1DOperator(NeighborhoodType &                  F,
                   const std::string &                 whichF,
                   const unsigned int                  WhichDimension,
                   const CoefficientImageSpacingType & spacing) const;

  /** Private function used for the filtering. It creates ND inseparable operators F. */
  void
  CreateNDOperator(NeighborhoodType & F, const std::string & whichF, const CoefficientImageSpacingType & spacing) const;

  /** Private function used for the filtering. It performs 1D separable filtering. */
  CoefficientImagePointer
  FilterSeparable(const CoefficientImageType *, const std::vector<NeighborhoodType> & Operators) const;

  /** Member variables. */
  BSplineTransformPointer m_BSplineTransform;
  ScalarType              m_LinearityConditionWeight;
  ScalarType              m_OrthonormalityConditionWeight;
  ScalarType              m_PropernessConditionWeight;

  mutable MeasureType m_RigidityPenaltyTermValue;
  mutable MeasureType m_LinearityConditionValue;
  mutable MeasureType m_OrthonormalityConditionValue;
  mutable MeasureType m_PropernessConditionValue;
  mutable MeasureType m_LinearityConditionGradientMagnitude;
  mutable MeasureType m_OrthonormalityConditionGradientMagnitude;
  mutable MeasureType m_PropernessConditionGradientMagnitude;

  bool m_UseLinearityCondition;
  bool m_UseOrthonormalityCondition;
  bool m_UsePropernessCondition;
  bool m_CalculateLinearityCondition;
  bool m_CalculateOrthonormalityCondition;
  bool m_CalculatePropernessCondition;

  /** Rigidity image variables. */
  CoordinateRepresentationType     m_DilationRadiusMultiplier;
  bool                             m_DilateRigidityImages;
  mutable bool                     m_RigidityCoefficientImageIsFilled;
  RigidityImagePointer             m_FixedRigidityImage;
  RigidityImagePointer             m_MovingRigidityImage;
  RigidityImagePointer             m_RigidityCoefficientImage;
  std::vector<DilateFilterPointer> m_FixedRigidityImageDilation;
  std::vector<DilateFilterPointer> m_MovingRigidityImageDilation;
  RigidityImagePointer             m_FixedRigidityImageDilated;
  RigidityImagePointer             m_MovingRigidityImageDilated;
  bool                             m_UseFixedRigidityImage;
  bool                             m_UseMovingRigidityImage;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTransformRigidityPenaltyTerm.hxx"
#endif

#endif // #ifndef itkTransformRigidityPenaltyTerm_h
