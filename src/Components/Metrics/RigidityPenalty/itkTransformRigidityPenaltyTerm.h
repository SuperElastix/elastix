/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkTransformRigidityPenaltyTerm_h
#define __itkTransformRigidityPenaltyTerm_h

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

template< class TFixedImage, class TScalarType >
class TransformRigidityPenaltyTerm
  : public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  /** Standard itk stuff. */
  typedef TransformRigidityPenaltyTerm    Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >            Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformRigidityPenaltyTerm, TransformPenaltyTerm );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer         MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImagePointer          FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::ImageSampleContainerType    ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer ImageSampleContainerPointer;
  typedef typename Superclass::ScalarType                 ScalarType;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int, FixedImageType::ImageDimension );
  itkStaticConstMacro( ImageDimension, unsigned int, FixedImageType::ImageDimension );

  /** Initialize the penalty term. */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Typedef's for B-spline transform. */
  typedef AdvancedBSplineDeformableTransform< ScalarType,
    FixedImageDimension, 3 >                            BSplineTransformType;
  typedef typename BSplineTransformType::Pointer        BSplineTransformPointer;
  typedef typename BSplineTransformType::SpacingType    GridSpacingType;
  typedef typename BSplineTransformType::ImageType      CoefficientImageType;
  typedef typename CoefficientImageType::Pointer        CoefficientImagePointer;
  typedef typename CoefficientImageType::SpacingType    CoefficientImageSpacingType;
  typedef AdvancedCombinationTransform< ScalarType,
    FixedImageDimension >                               CombinationTransformType;

  /** Typedef support for neighborhoods, filters, etc. */
  typedef Neighborhood< ScalarType,
    itkGetStaticConstMacro( FixedImageDimension ) >     NeighborhoodType;
  typedef typename NeighborhoodType::SizeType           NeighborhoodSizeType;
  typedef ImageRegionIterator< CoefficientImageType >   CoefficientImageIteratorType;
  typedef NeighborhoodOperatorImageFilter<
    CoefficientImageType, CoefficientImageType >        NOIFType;
  typedef NeighborhoodIterator<CoefficientImageType>    NeighborhoodIteratorType;
  typedef typename NeighborhoodIteratorType::RadiusType RadiusType;

  /** Typedef's for the construction of the rigidity image. */
  typedef CoefficientImageType                          RigidityImageType;
  typedef typename RigidityImageType::Pointer           RigidityImagePointer;
  typedef typename RigidityImageType::PixelType         RigidityPixelType;
  typedef typename RigidityImageType::RegionType        RigidityImageRegionType;
  typedef typename RigidityImageType::IndexType         RigidityImageIndexType;
  typedef typename RigidityImageType::PointType         RigidityImagePointType;
  typedef ImageRegionIterator< RigidityImageType >      RigidityImageIteratorType;
  typedef BinaryBallStructuringElement<
    RigidityPixelType,
    itkGetStaticConstMacro( FixedImageDimension ) >     StructuringElementType;
  typedef typename StructuringElementType::RadiusType   SERadiusType;
  typedef GrayscaleDilateImageFilter<
    RigidityImageType, RigidityImageType,
    StructuringElementType >                            DilateFilterType;
  typedef typename DilateFilterType::Pointer            DilateFilterPointer;

  /** Check stuff. */
  void CheckUseAndCalculationBooleans( void );

  /** The GetValue()-method returns the rigid penalty value. */
  virtual MeasureType GetValue(
    const ParametersType & parameters ) const;

  /** The GetDerivative()-method returns the rigid penalty derivative. */
  virtual void GetDerivative(
    const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Contains calls from GetValueAndDerivative that are thread-unsafe. */
  virtual void BeforeThreadedGetValueAndDerivative(
    const TransformParametersType & parameters ) const;

  /** The GetValueAndDerivative()-method returns the rigid penalty value and its derivative. */
  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  /** Set the B-spline transform in this class.
   * This class expects a BSplineTransform! It is not suited for others.
   */
  itkSetObjectMacro( BSplineTransform, BSplineTransformType );

  /** Set the RigidityImage in this class. */
  //itkSetObjectMacro( RigidityCoefficientImage, RigidityImageType );

  /** Set/Get the weight of the linearity condition part. */
  itkSetClampMacro( LinearityConditionWeight, ScalarType,
    0.0, NumericTraits<ScalarType>::max() );
  itkGetMacro( LinearityConditionWeight, ScalarType );

  /** Set/Get the weight of the orthonormality condition part. */
  itkSetClampMacro( OrthonormalityConditionWeight, ScalarType,
    0.0, NumericTraits<ScalarType>::max() );
  itkGetMacro( OrthonormalityConditionWeight, ScalarType );

  /** Set/Get the weight of the properness condition part. */
  itkSetClampMacro( PropernessConditionWeight, ScalarType,
    0.0, NumericTraits<ScalarType>::max() );
  itkGetMacro( PropernessConditionWeight, ScalarType );

  /** Set the usage of the linearity condition part. */
  itkSetMacro( UseLinearityCondition, bool );

  /** Set the usage of the orthonormality condition part. */
  itkSetMacro( UseOrthonormalityCondition, bool );

  /** Set the usage of the properness condition part. */
  itkSetMacro( UsePropernessCondition, bool );

  /** Set the calculation of the linearity condition part,
   * even if we don't use it.
   */
  itkSetMacro( CalculateLinearityCondition, bool );

  /** Set the calculation of the orthonormality condition part,
   * even if we don't use it.
   */
  itkSetMacro( CalculateOrthonormalityCondition, bool );

  /** Set the calculation of the properness condition part.,
   * even if we don't use it.
   */
  itkSetMacro( CalculatePropernessCondition, bool );

  /** Get the value of the linearity condition. */
  itkGetConstReferenceMacro( LinearityConditionValue, MeasureType );

  /** Get the value of the orthonormality condition. */
  itkGetConstReferenceMacro( OrthonormalityConditionValue, MeasureType );

  /** Get the value of the properness condition. */
  itkGetConstReferenceMacro( PropernessConditionValue, MeasureType );

  /** Get the gradient magnitude of the linearity condition. */
  itkGetConstReferenceMacro( LinearityConditionGradientMagnitude, MeasureType );

  /** Get the gradient magnitude of the orthonormality condition. */
  itkGetConstReferenceMacro( OrthonormalityConditionGradientMagnitude, MeasureType );

  /** Get the gradient magnitude of the properness condition. */
  itkGetConstReferenceMacro( PropernessConditionGradientMagnitude, MeasureType );

  /** Get the value of the total rigidity penalty term. */
  //itkGetConstReferenceMacro( RigidityPenaltyTermValue, MeasureType );

  /** Set if the RigidityImage's are dilated. */
  itkSetMacro( DilateRigidityImages, bool );

  /** Set the DilationRadiusMultiplier. */
  itkSetClampMacro( DilationRadiusMultiplier, CoordinateRepresentationType,
    0.1, NumericTraits<CoordinateRepresentationType>::max() );

  /** Set the fixed coefficient image. */
  itkSetObjectMacro( FixedRigidityImage, RigidityImageType );

  /** Set the moving coefficient image. */
  itkSetObjectMacro( MovingRigidityImage, RigidityImageType );

  /** Set to use the FixedRigidityImage or not. */
  itkSetMacro( UseFixedRigidityImage, bool );

  /** Set to use the MovingRigidityImage or not. */
  itkSetMacro( UseMovingRigidityImage, bool );

  /** Function to fill the RigidityCoefficientImage every iteration. */
  void FillRigidityCoefficientImage( const ParametersType & parameters ) const;

protected:

  /** The constructor. */
  TransformRigidityPenaltyTerm();
  /** The destructor. */
  virtual ~TransformRigidityPenaltyTerm() {};

  /** PrintSelf. */
  void PrintSelf( std::ostream& os, Indent indent ) const;

private:

  /** The private constructor. */
  TransformRigidityPenaltyTerm( const Self& ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self& );            // purposely not implemented

  /** Internal function to dilate the rigidity images. */
  virtual void DilateRigidityImages( void );

  /** Private function used for the filtering. It creates 1D separable operators F. */
  void Create1DOperator( NeighborhoodType & F, const std::string WhichF,
    const unsigned int WhichDimension, const CoefficientImageSpacingType & spacing ) const;

  /** Private function used for the filtering. It creates ND inseparable operators F. */
  void CreateNDOperator( NeighborhoodType & F, const std::string WhichF,
    const CoefficientImageSpacingType & spacing ) const;

  /** Private function used for the filtering. It performs 1D separable filtering. */
  CoefficientImagePointer FilterSeparable( const CoefficientImageType *,
    const std::vector< NeighborhoodType > &Operators ) const;

  /** Member variables. */
  BSplineTransformPointer m_BSplineTransform;
  ScalarType              m_LinearityConditionWeight;
  ScalarType              m_OrthonormalityConditionWeight;
  ScalarType              m_PropernessConditionWeight;

  mutable MeasureType     m_RigidityPenaltyTermValue;
  mutable MeasureType     m_LinearityConditionValue;
  mutable MeasureType     m_OrthonormalityConditionValue;
  mutable MeasureType     m_PropernessConditionValue;
  mutable MeasureType     m_LinearityConditionGradientMagnitude;
  mutable MeasureType     m_OrthonormalityConditionGradientMagnitude;
  mutable MeasureType     m_PropernessConditionGradientMagnitude;

  bool                    m_UseLinearityCondition;
  bool                    m_UseOrthonormalityCondition;
  bool                    m_UsePropernessCondition;
  bool                    m_CalculateLinearityCondition;
  bool                    m_CalculateOrthonormalityCondition;
  bool                    m_CalculatePropernessCondition;

  /** Rigidity image variables. */
  CoordinateRepresentationType    m_DilationRadiusMultiplier;
  bool                            m_DilateRigidityImages;
  mutable bool                    m_RigidityCoefficientImageIsFilled;
  RigidityImagePointer            m_FixedRigidityImage;
  RigidityImagePointer            m_MovingRigidityImage;
  RigidityImagePointer            m_RigidityCoefficientImage;
  std::vector< DilateFilterPointer >  m_FixedRigidityImageDilation;
  std::vector< DilateFilterPointer >  m_MovingRigidityImageDilation;
  RigidityImagePointer            m_FixedRigidityImageDilated;
  RigidityImagePointer            m_MovingRigidityImageDilated;
  bool                            m_UseFixedRigidityImage;
  bool                            m_UseMovingRigidityImage;

}; // end class TransformRigidityPenaltyTerm


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTransformRigidityPenaltyTerm.txx"
#endif

#endif // #ifndef __itkTransformRigidityPenaltyTerm_h

