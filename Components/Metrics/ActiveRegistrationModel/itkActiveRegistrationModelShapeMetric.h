/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkActiveRegistrationModelShapeMetric_h__
#define __itkActiveRegistrationModelShapeMetric_h__

#include "itkSingleValuedPointSetToPointSetMetric.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"
#include "itkMesh.h"
#include <itkVectorContainer.h>
#include <string>

#include "itkDataManager.h"
#include "itkStatisticalModel.h"
#include "itkPCAModelBuilder.h"
#include "itkReducedVarianceModelBuilder.h"
#include "itkStandardMeshRepresenter.h"

namespace itk
{

/** \class PointSetPenalty
 * \brief A dummy metric to generate transformed meshes each iteration.
 *
 *
 *
 * \ingroup RegistrationMetrics
 */

template< class TFixedPointSet, class TMovingPointSet >
class ITK_EXPORT ActiveRegistrationModelShapeMetric :
  public SingleValuedPointSetToPointSetMetric< TFixedPointSet, TMovingPointSet >
{
public:

  /** Standard class typedefs. */
  typedef ActiveRegistrationModelShapeMetric                 Self;
  typedef SingleValuedPointSetToPointSetMetric<
    TFixedPointSet, TMovingPointSet > Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

  /** Type used for representing point components  */

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( PointDistributionShapeMetric, SingleValuedPointSetToPointSetMetric );

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType           TransformType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;

  typedef typename Superclass::MeasureType         MeasureType;
  typedef typename Superclass::DerivativeType      DerivativeType;
  typedef typename Superclass::DerivativeValueType DerivativeValueType;

  /** Typedefs. */
  typedef typename Superclass::InputPointType         InputPointType;
  typedef typename Superclass::OutputPointType        OutputPointType;
  typedef typename InputPointType::CoordRepType       CoordRepType;
  typedef vnl_vector<CoordRepType>                    VnlVectorType;
  typedef typename TransformType::InputPointType      FixedImagePointType;
  typedef typename TransformType::OutputPointType     MovingImagePointType;
  typedef typename TransformType::SpatialJacobianType SpatialJacobianType;

  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Constants for the pointset dimensions. */
  itkStaticConstMacro( FixedPointSetDimension, unsigned int,
    Superclass::FixedPointSetDimension );

  typedef Vector< typename TransformType::ScalarType,
    FixedPointSetDimension >                                          PointNormalType;
  typedef unsigned char DummyMeshPixelType;
  typedef DefaultStaticMeshTraits< PointNormalType,
    FixedPointSetDimension, FixedPointSetDimension, CoordRepType >    MeshTraitsType;
  typedef Mesh< PointNormalType, FixedPointSetDimension,
    MeshTraitsType >                                                  FixedMeshType;

  typedef typename FixedMeshType::ConstPointer             FixedMeshConstPointer;
  typedef typename FixedMeshType::Pointer                  FixedMeshPointer;
  typedef typename MeshTraitsType::CellType::CellInterface CellInterfaceType;

  // ActiveRegistrationModel typedefs
  typedef double                                                                  StatisticalModelScalarType;
  typedef vnl_vector< double >                                                    StatisticalModelVectorType;
  typedef vnl_matrix< double  >                                                   StatisticalModelMatrixType;

  itkStaticConstMacro( StatisticalModelMeshDimension, unsigned int, Superclass::FixedPointSetDimension );

  typedef DefaultStaticMeshTraits<
    StatisticalModelScalarType,
    FixedPointSetDimension,
    FixedPointSetDimension,
    StatisticalModelScalarType,
    StatisticalModelScalarType >                                                  StatisticalModelMeshTraitsType;

  typedef Mesh<
    StatisticalModelScalarType,
    StatisticalModelMeshDimension,
    StatisticalModelMeshTraitsType >                                              StatisticalModelMeshType;
  typedef typename StatisticalModelMeshType::Pointer                              StatisticalModelMeshPointer;
  typedef typename StatisticalModelMeshType::ConstPointer                         StatisticalModelMeshConstPointer;
  typedef typename StatisticalModelMeshType::PointsContainerIterator              StatisticalModelMeshIteratorType;
  typedef typename StatisticalModelMeshType::PointsContainerConstIterator         StatisticalModelMeshConstIteratorType;

  typedef MeshFileReader< StatisticalModelMeshType >                              MeshReaderType;
  typedef typename MeshReaderType::Pointer                                        MeshReaderPointer;

  typedef DataManager< StatisticalModelMeshType >                                 StatisticalModelDataManagerType;
  typedef typename StatisticalModelDataManagerType::Pointer                       StatisticalModelDataManagerPointer;

  typedef StatisticalModel< StatisticalModelMeshType >                            StatisticalModelType;
  typedef typename StatisticalModelType::Pointer                                  StatisticalModelPointer;

  typedef StandardMeshRepresenter<
          StatisticalModelScalarType,
          StatisticalModelMeshDimension >                                         StatisticalModelRepresenterType;
  typedef typename StatisticalModelRepresenterType::Pointer                       StatisticalModelRepresenterPointer;

  typedef PCAModelBuilder< StatisticalModelMeshType >                             ModelBuilderType;
  typedef typename ModelBuilderType::Pointer                                      ModelBuilderPointer;

  typedef ReducedVarianceModelBuilder< StatisticalModelMeshType >                 StatisticalModelReducedVarianceBuilderType;
  typedef typename StatisticalModelReducedVarianceBuilderType::Pointer            StatisticalModelReducedVarianceBuilderPointer;

  typedef unsigned int                                                            StatisticalModelIdType;

  typedef VectorContainer< StatisticalModelIdType, StatisticalModelPointer >      StatisticalModelContainerType;
  typedef typename StatisticalModelContainerType::Pointer                         StatisticalModelContainerPointer;
  typedef typename StatisticalModelContainerType::ConstPointer                    StatisticalModelContainerConstPointer;

  itkSetConstObjectMacro( StatisticalModelContainer, StatisticalModelContainerType );
  itkGetConstObjectMacro( StatisticalModelContainer, StatisticalModelContainerType );

  /** Initialize the Metric by making sure that all the components are
  *  present and plugged together correctly.
  */
  virtual void Initialize( void );

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType& Derivative ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType& parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  void GetValueAndFiniteDifferenceDerivative( const TransformParametersType& parameters,
                                              MeasureType& value,
                                              DerivativeType& derivative ) const;

  void GetModelValue( const TransformParametersType& parameters,
                      const StatisticalModelPointer statisticalModel,
                      MeasureType& modelValue ) const;


  void GetModelFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                           const StatisticalModelPointer statisticalModel,
                                           DerivativeType& modelDerivative ) const;

protected:

  ActiveRegistrationModelShapeMetric();
  virtual ~ActiveRegistrationModelShapeMetric();

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  ActiveRegistrationModelShapeMetric( const Self & );    // purposely not implemented
  void operator=( const Self & ); // purposely not implemented

  StatisticalModelMeshPointer TransformMesh( StatisticalModelMeshPointer fixedMesh ) const;

  StatisticalModelContainerConstPointer m_StatisticalModelContainer;

}; // end class PointSetPenalty

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkActiveRegistrationModelShapeMetric.hxx"
#endif

#endif // end #ifndef __itkActiveRegistrationModelPointDistributionShapeMetric_h__

