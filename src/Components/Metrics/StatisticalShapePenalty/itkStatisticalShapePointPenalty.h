/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

  If you use the StatisticalShapePenalty anywhere we would appreciate if you cite the following article:
  F.F. Berendsen et al., Free-form image registration regularized by a statistical shape model: application to organ segmentation in cervical MR, Comput. Vis. Image Understand. (2013), http://dx.doi.org/10.1016/j.cviu.2012.12.006

======================================================================*/
#ifndef __itkStatisticalShapePointPenalty_h
#define __itkStatisticalShapePointPenalty_h

#include "itkSingleValuedPointSetToPointSetMetric.h"

#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"
#include "itkArray.h"

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_math.h>
#include <vnl/vnl_vector.h>
#include <vnl/algo/vnl_real_eigensystem.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
//#include <vnl/algo/vnl_svd.h>
#include <vnl/algo/vnl_svd_economy.h>

#include <itkVariableSizeMatrix.h>

#include <vcl_iostream.h>
#include <string>



namespace itk
{

/** \class StatisticalShapePointPenalty
 * \brief Computes the Mahalanobis distance between the transformed shape and a mean shape.
 *  A model mean and covariance are required.
 *
 *
 * \ingroup RegistrationMetrics
 */

template < class TFixedPointSet, class TMovingPointSet >
class ITK_EXPORT StatisticalShapePointPenalty :
    public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
{
public:

  /** Standard class typedefs. */
  typedef StatisticalShapePointPenalty    Self;
  typedef SingleValuedPointSetToPointSetMetric<
    TFixedPointSet, TMovingPointSet >               Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StatisticalShapePointPenalty,
    SingleValuedPointSetToPointSetMetric );

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::FixedPointSetType          FixedPointSetType;
  typedef typename Superclass::MovingPointSetType         MovingPointSetType;
  typedef typename Superclass::FixedPointSetConstPointer  FixedPointSetConstPointer;
  typedef typename Superclass::MovingPointSetConstPointer MovingPointSetConstPointer;

  typedef typename Superclass::PointIterator              PointIterator;
  typedef typename Superclass::PointDataIterator          PointDataIterator;

  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;

  typedef typename OutputPointType::CoordRepType          CoordRepType;
  typedef vnl_vector<CoordRepType>               VnlVectorType;
  typedef vnl_matrix<CoordRepType>               VnlMatrixType;
  //typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  //typedef itk::Array<VnlVectorType *> ProposalDerivativeType;
  typedef typename std::vector< VnlVectorType *> ProposalDerivativeType;
  //typedef typename vnl_vector<VnlVectorType *> ProposalDerivativeType; //Cannot be linked
  typedef vnl_svd_economy<CoordRepType> PCACovarianceType;

  void Initialize( void ) throw ( ExceptionObject );

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & Derivative ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set/Get if the distance should be squared. Default is true for computation speed */

  /** Set/Get the shrinkageIntensity parameter. */
  itkSetClampMacro( ShrinkageIntensity, MeasureType,
    0.0, 1.0 );
  itkGetMacro( ShrinkageIntensity, MeasureType );
  
  itkSetMacro(ShrinkageIntensityNeedsUpdate, bool);
  itkBooleanMacro( ShrinkageIntensityNeedsUpdate );

    /** Set/Get the BaseVariance parameter. */
  itkSetClampMacro( BaseVariance, MeasureType,
    -1.0, NumericTraits<MeasureType>::max() );
  itkGetMacro( BaseVariance, MeasureType );
 
  itkSetMacro(BaseVarianceNeedsUpdate, bool);
  itkBooleanMacro( BaseVarianceNeedsUpdate );

  itkSetClampMacro( CentroidXVariance, MeasureType,
    -1.0, NumericTraits<MeasureType>::max() );
  itkGetMacro( CentroidXVariance, MeasureType );

  itkSetClampMacro( CentroidYVariance, MeasureType,
    -1.0, NumericTraits<MeasureType>::max() );
  itkGetMacro( CentroidYVariance, MeasureType );

  itkSetClampMacro( CentroidZVariance, MeasureType,
    -1.0, NumericTraits<MeasureType>::max() );
  itkGetMacro( CentroidZVariance, MeasureType );

  itkSetClampMacro( SizeVariance, MeasureType,
    -1.0, NumericTraits<MeasureType>::max() );
  itkGetMacro( SizeVariance, MeasureType );

  itkSetMacro(VariancesNeedsUpdate, bool);
  itkBooleanMacro( VariancesNeedsUpdate );

  itkSetClampMacro( CutOffValue, MeasureType,
   0.0, NumericTraits<MeasureType>::max() );
  itkGetMacro( CutOffValue, MeasureType );

  itkSetClampMacro( CutOffSharpness, MeasureType,
   NumericTraits<MeasureType>::NonpositiveMin(), NumericTraits<MeasureType>::max() );
  itkGetMacro( CutOffSharpness, MeasureType );

  /** Set/Get the EigenVectorName parameter. */
  //itkSetStringMacro(EigenVectorsName);
  //itkGetStringMacro(EigenVectorsName);

  /** Set/Get the EigenValuesName parameter. */
  //itkSetStringMacro(EigenValuesName);
  //itkGetStringMacro(EigenValuesName);
  itkSetMacro(ShapeModelCalculation, int);
  itkGetConstReferenceMacro( ShapeModelCalculation, int );

  itkSetMacro(NormalizedShapeModel, bool);
  itkGetConstReferenceMacro( NormalizedShapeModel, bool );
  itkBooleanMacro( NormalizedShapeModel );

  itkSetConstObjectMacro(EigenVectors,vnl_matrix<double>);
  itkSetConstObjectMacro(EigenValues,vnl_vector<double>);
  itkSetConstObjectMacro(MeanVector,vnl_vector<double>);
  
  itkSetConstObjectMacro(CovarianceMatrix,vnl_matrix<double>);
  //void SetEigenVectors(vnl_matrix<double>*);
  //void SetEigenValues(vnl_vector<double>*);


protected:
  StatisticalShapePointPenalty();
  virtual ~StatisticalShapePointPenalty();

  /** PrintSelf. */
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  StatisticalShapePointPenalty(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void FillProposalVector(const OutputPointType & fixedPoint, const unsigned int vertexindex) const;
  void FillProposalDerivative(const OutputPointType & fixedPoint, const unsigned int vertexindex) const;

  void UpdateCentroidAndAlignProposalVector(const unsigned int shapeLength) const;
  void UpdateCentroidAndAlignProposalDerivative(const unsigned int shapeLength) const;

  void UpdateL2(const unsigned int shapeLength) const;
  void NormalizeProposalVector(const unsigned int shapeLength) const;
  void UpdateL2AndNormalizeProposalDerivative(const unsigned int shapeLength) const;

  void CalculateValue(MeasureType & value , VnlVectorType & differenceVector , VnlVectorType & centerrotated , VnlVectorType & eigrot) const;
  void CalculateDerivative(DerivativeType & derivative , const MeasureType & value , const VnlVectorType & differenceVector , const VnlVectorType & centerrotated , const VnlVectorType & eigrot, const unsigned int shapeLength) const;
   
  void CalculateCutOffValue( MeasureType & value) const;

  void CalculateCutOffDerivative( typename DerivativeType::element_type & derivativeElement , const MeasureType & value ) const;
     
	  //vnl_matrix<double> m_invCovariance;
  //std::string m_EigenVectorsName;  
  //std::string m_EigenValuesName;

    //vnl_matrix<double> m_eigenvectors;
    //vnl_vector<double> m_eigenvalues;
    const VnlVectorType* m_MeanVector;
    const VnlMatrixType* m_CovarianceMatrix;
    const VnlMatrixType* m_EigenVectors;
    const VnlVectorType* m_EigenValues;

    VnlMatrixType* m_InverseCovarianceMatrix;

    double m_CentroidXVariance;
    double m_CentroidXStd;
    double m_CentroidYVariance;
    double m_CentroidYStd;
    double m_CentroidZVariance;
    double m_CentroidZStd;
    double m_SizeVariance;
    double m_SizeStd;

    bool m_ShrinkageIntensityNeedsUpdate;
    bool m_BaseVarianceNeedsUpdate;
    bool m_VariancesNeedsUpdate;

    VnlVectorType* m_EigenValuesRegularized;
  
    //mutable ProposalDerivativeType m_ProposalDerivative;
    mutable ProposalDerivativeType* m_ProposalDerivative;
    //ProposalDerivativeType::Pointer m_ProposalDerivative;
    //mutable PCACovarianceType* m_PCACovariance;
    unsigned int m_ProposalLength;
    bool m_NormalizedShapeModel;
    int m_ShapeModelCalculation;
    double m_ShrinkageIntensity;
    double m_BaseVariance;
    double m_BaseStd;
    mutable VnlVectorType m_proposalvector;
    mutable VnlVectorType m_meanvalues;

    double m_CutOffValue;
    double m_CutOffSharpness;
    /*
    mutable VnlMatrixType m_eigenvectors;
    mutable VnlVectorType m_eigenvalues;
    mutable VnlVectorType m_proposalvector;
    mutable VnlVectorType m_meanvalues;
*/
}; // end class StatisticalShapePointPenalty

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStatisticalShapePointPenalty.txx"
#endif

#endif
