/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkRigidityPenaltyTermMetric_h
#define __itkRigidityPenaltyTermMetric_h

#include "itkSingleValuedCostFunction.h"

#include "itkBSplineDeformableTransform.h"

#include "itkNeighborhood.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkNeighborhoodIterator.h"


namespace itk
{
  /** 
   * \class RigidityPenaltyTermMetric
   * \brief A metric that calculates a rigidity penalty term
   * on the B-spline coefficients of a B-spline transformation.
   * This penalty term is a function of the 1st and 2nd order spatial
   * derivatives of a transformation.
   *
   * The intended use for this metric is to filter a B-spline coefficient
   * image in order to calculate a rigidity penalty term on a B-spline transform. 
   * 
   * \par
   * The RigidityPenaltyTermValueImageFilter at each pixel location is computed by
   * convolution with some separable 1D kernels.
   *
   * The rigid penalty term penalizes deviations from a rigid
   * transformation at regions specified by the so-called rigidity images.
   *
   * This metric only works with B-splines as a transformation model.
   *
   * References:
   * [1] "Nonrigid Registration Using a Rigidity Constraint"
   *    M. Staring, S. Klein and Josien P.W. Pluim
   *    SPIE Medical Imaging 2006: Image Processing, 2006.
   * 
   * \sa BSplineTransform
   * \sa MattesMutualInformationImageToImageMetricWithRigidRegularization
   * \sa MattesMutualInformationMetricWithRigidRegularization
   * \ingroup Metrics
   */

  template< unsigned int Dimension, class TScalarType >
  class RigidityPenaltyTermMetric : public SingleValuedCostFunction
  {
  public:

    /** Standard itk stuff. */
    typedef RigidityPenaltyTermMetric       Self;
    typedef SingleValuedCostFunction        Superclass;
    typedef SmartPointer<Self>              Pointer;
    typedef SmartPointer<const Self>        ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( RigidityPenaltyTermMetric, SingleValuedCostFunction );

    /** Typedef's inherited from the superclass. */
    typedef typename Superclass::MeasureType         MeasureType;
    typedef typename Superclass::DerivativeType      DerivativeType;
    typedef typename Superclass::ParametersType      ParametersType;

    /** Template parameters. */
    typedef TScalarType   ScalarType;

    /** Define the dimension. */
    itkStaticConstMacro( ImageDimension, unsigned int, Dimension );

    /** Typedef's for BSpline transform. */
    typedef BSplineDeformableTransform< ScalarType,
      Dimension, 3 >                                      BSplineTransformType;
    typedef typename BSplineTransformType::Pointer        BSplineTransformPointer;
    typedef typename BSplineTransformType::ImageType      CoefficientImageType;
    typedef typename CoefficientImageType::Pointer        CoefficientImagePointer;
    typedef typename CoefficientImageType::SpacingType    CoefficientImageSpacingType;

    /** Typedef support for neigborhoods, filters, etc. */
    typedef Neighborhood< ScalarType,
      itkGetStaticConstMacro( ImageDimension ) >          NeighborhoodType;
    typedef typename NeighborhoodType::SizeType           NeighborhoodSizeType;
    typedef ImageRegionIterator< CoefficientImageType >   CoefficientImageIteratorType;
    typedef NeighborhoodOperatorImageFilter<
      CoefficientImageType, CoefficientImageType >        NOIFType;
    typedef NeighborhoodIterator<CoefficientImageType>    NeighborhoodIteratorType;
    typedef typename NeighborhoodIteratorType::RadiusType RadiusType;
    
    /** Check stuff. */
    void CheckUseAndCalculationBooleans( void );

    /** The GetValue()-method returns the rigid penalty value. */
    virtual MeasureType GetValue(
      const ParametersType & parameters ) const;

    /** The GetDerivative()-method returns the rigid penalty derivative. */
    virtual void GetDerivative(
      const ParametersType & parameters,
      DerivativeType & derivative ) const;

    /** The GetValueAndDerivative()-method returns the rigid penalty value and its derivative. */
    virtual void GetValueAndDerivative(
      const ParametersType & parameters,
      MeasureType & value,
      DerivativeType & derivative ) const;

    /** Get the number of parameters. */
    virtual unsigned int GetNumberOfParameters(void) const;

    /** Set the BSpline transform in this class.
     * This class expects a BSplineTransform! It is not suited for others.
     */
    itkSetObjectMacro( BSplineTransform, BSplineTransformType );

    /** Set the RigidityImage in this class. */
    itkSetObjectMacro( RigidityCoefficientImage, CoefficientImageType );

    /** Set/Get the weight of the linearity condition part. */
    itkSetClampMacro( LinearityConditionWeight, ScalarType,
      0.0, NumericTraits<ScalarType>::max() );
    itkGetMacro( LinearityConditionWeight, ScalarType );

    /** Set/Get the weight of the orthonormality condition part. */
    itkSetClampMacro( OrthonormalityConditionWeight, ScalarType,
      0.0, NumericTraits<ScalarType>::max() );
    itkGetMacro( OrthonormalityConditionWeight, ScalarType );

    /** Set/Get the weight of the properness condition part: the incompressibility condition. */
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

    /** Get the value of the total rigidity penalty term. */
    itkGetConstReferenceMacro( RigidityPenaltyTermValue, MeasureType );

  protected:

    /** The constructor. */
    RigidityPenaltyTermMetric();
    /** The destructor. */
    virtual ~RigidityPenaltyTermMetric() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

  private:

    /** The private constructor. */
    RigidityPenaltyTermMetric( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );            // purposely not implemented

    /** Private function used for the filtering. It creates 1D separable operators F. */
    void Create1DOperator( NeighborhoodType & F, const std::string WhichF,
      const unsigned int WhichDimension, const CoefficientImageSpacingType & spacing ) const;

    /** Private function used for the filtering. It creates ND unseparable operators F. */
    void CreateNDOperator( NeighborhoodType & F, const std::string WhichF,
      const CoefficientImageSpacingType & spacing ) const;

    /** Private function used for the filtering. It performs 1D separable filtering. */
    CoefficientImagePointer FilterSeparable( const CoefficientImageType *,
      const std::vector< NeighborhoodType > &Operators ) const;

    /** Member variables. */
    BSplineTransformPointer     m_BSplineTransform;
    CoefficientImagePointer     m_RigidityCoefficientImage;

    /** A private variable to store the weighting of the linearity condition. */
    ScalarType  m_LinearityConditionWeight;

    /** A private variable to store the weighting of the orthonormality condition. */
    ScalarType  m_OrthonormalityConditionWeight;

    /** A private variable to store the weighting of the properness condition. */
    ScalarType  m_PropernessConditionWeight;

    /** Private variables to store the rigidity metric values. */
    mutable MeasureType     m_RigidityPenaltyTermValue;
    mutable MeasureType     m_LinearityConditionValue;
    mutable MeasureType     m_OrthonormalityConditionValue;
    mutable MeasureType     m_PropernessConditionValue;

    /** Name of the output directory. */
    //std::string m_OutputDirectoryName;

    /** Variables to store the use and calculation of
     * the different conditions.
     */
    bool m_UseLinearityCondition;
    bool m_UseOrthonormalityCondition;
    bool m_UsePropernessCondition;
    bool m_CalculateLinearityCondition;
    bool m_CalculateOrthonormalityCondition;
    bool m_CalculatePropernessCondition;

  }; // end class RigidityPenaltyTermMetric
  

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRigidityPenaltyTermMetric.txx"
#endif

#endif // #ifndef __itkRigidityPenaltyTermMetric_h

