/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkActiveRegistrationModelPointDistributionShapeMetric_h__
#define __itkActiveRegistrationModelPointDistributionShapeMetric_h__

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

  typedef typename FixedMeshType::PointType             MeshPointType;
  typedef typename FixedMeshType::PointType::VectorType VectorType;

  typedef typename FixedMeshType::PointsContainer              MeshPointsContainerType;
  typedef typename MeshPointsContainerType::Pointer            MeshPointsContainerPointer;
  typedef typename MeshPointsContainerType::ConstPointer       MeshPointsContainerConstPointer;
  typedef typename FixedMeshType::PointsContainerConstIterator MeshPointsContainerConstIteratorType;
  typedef typename FixedMeshType::PointsContainerIterator      MeshPointsContainerIteratorType;

  typedef typename FixedMeshType::PointDataContainer             MeshPointDataContainerType;
  typedef typename FixedMeshType::PointDataContainerConstPointer MeshPointDataContainerConstPointer;
  typedef typename FixedMeshType::PointDataContainerPointer      MeshPointDataContainerPointer;
  //typedef typename FixedMeshType::PointDataContainerConstIterator     MeshPointDataContainerConstIteratorType;
  typedef typename FixedMeshType::PointDataContainerIterator MeshPointDataContainerConstIteratorType;
  typedef typename MeshPointDataContainerType::Iterator      MeshPointDataContainerIteratorType;

  typedef unsigned int                                          MeshIdType;
  typedef VectorContainer< MeshIdType, FixedMeshConstPointer >  FixedMeshContainerType;
  typedef typename FixedMeshContainerType::Pointer              FixedMeshContainerPointer;
  typedef typename FixedMeshContainerType::ConstPointer         FixedMeshContainerConstPointer;
  typedef typename FixedMeshContainerType::ElementIdentifier    FixedMeshContainerElementIdentifier;

  typedef VectorContainer< MeshIdType, FixedMeshPointer >   MappedMeshContainerType;
  typedef typename MappedMeshContainerType::Pointer         MappedMeshContainerPointer;
  typedef typename MappedMeshContainerType::ConstPointer    MappedMeshContainerConstPointer;

  typedef Array< DerivativeValueType > MeshPointsDerivativeValueType;

  // ActiveRegistrationModel typedefs
  typedef double                                                                  StatisticalModelCoordRepType;
  typedef vnl_vector< float >                                                     StatisticalModelFloatVectorType;
  typedef vnl_vector< double >                                                    StatisticalModelVectorType;
  typedef vnl_vector< double >                                                    StatisticalModelDoublePrecisionVectorType;
  typedef vnl_matrix< double  >                                                   StatisticalModelMatrixType;
  typedef double                                                                  StatisticalModelMeasureType;
  typedef vnl_vector< double >                                                    StatisticalModelDerivativeType;

  itkStaticConstMacro( StatisticalModelMeshDimension, unsigned int, Superclass::FixedPointSetDimension );

  typedef DefaultStaticMeshTraits<
    StatisticalModelCoordRepType,
    FixedPointSetDimension,
    FixedPointSetDimension,
    StatisticalModelCoordRepType,
    StatisticalModelCoordRepType >                                                StatisticalModelMeshTraitsType;

  typedef Mesh<
    StatisticalModelCoordRepType,
    StatisticalModelMeshDimension,
    StatisticalModelMeshTraitsType >                                              StatisticalModelMeshType;

  typedef typename StatisticalModelMeshType::PointType                            StatisticalModelPointType;
  typedef typename StatisticalModelMeshType::Pointer                              StatisticalModelMeshPointer;
  typedef typename StatisticalModelMeshType::ConstPointer                         StatisticalModelMeshConstPointer;
  typedef typename StatisticalModelMeshType::PointsContainerIterator              StatisticalModelMeshIteratorType;
  typedef typename StatisticalModelMeshType::PointsContainerConstIterator         StatisticalModelMeshConstIteratorType;

  typedef MeshFileReader< StatisticalModelMeshType >                              MeshReaderType;
  typedef typename MeshReaderType::Pointer                                        MeshReaderPointer;

  typedef vnl_vector< double >                                                    StatisticalModelParameterVectorType;
  typedef std::vector< std::string >                                              StatisticalModelPathVectorType;

  typedef StandardMeshRepresenter<
    StatisticalModelCoordRepType,
    StatisticalModelMeshDimension >                                               RepresenterType;
  typedef typename RepresenterType::Pointer                                       RepresenterPointer;

  typedef DataManager< StatisticalModelMeshType >                                 DataManagerType;
  typedef typename DataManagerType::Pointer                                       DataManagerPointer;

  typedef StatisticalModel< StatisticalModelMeshType >                            StatisticalModelType;
  typedef typename StatisticalModelType::Pointer                                  StatisticalModelPointer;
  typedef typename StatisticalModelType::ConstPointer                             StatisticalModelConstPointer;
  typedef typename StatisticalModelType::PointValuePairType                       StatisticalModelPointValuePairType;
  typedef typename StatisticalModelType::PointValueListType                       StatisticalModelPointValueListType;

  typedef PCAModelBuilder< StatisticalModelMeshType >                             ModelBuilderType;
  typedef typename ModelBuilderType::Pointer                                      ModelBuilderPointer;

  typedef ReducedVarianceModelBuilder< StatisticalModelMeshType >                 ReducedVarianceModelBuilderType;
  typedef typename ReducedVarianceModelBuilderType::Pointer                       ReducedVarianceModelBuilderPointer;

  typedef unsigned int                                                            StatisticalModelIdType;
  typedef VectorContainer< StatisticalModelIdType, StatisticalModelConstPointer > StatisticalModelContainerType;
  typedef typename StatisticalModelContainerType::Pointer                         StatisticalModelContainerPointer;
  typedef typename StatisticalModelContainerType::ConstPointer                    StatisticalModelContainerConstPointer;
  typedef typename StatisticalModelContainerType::ConstIterator                   StatisticalModelContainerConstIterator;

  typedef VectorContainer< StatisticalModelIdType, StatisticalModelMatrixType >   StatisticalModelMatrixContainerType;
  typedef typename StatisticalModelMatrixContainerType::Pointer                   StatisticalModelMatrixContainerPointer;
  typedef typename StatisticalModelMatrixContainerType::ConstPointer              StatisticalModelMatrixContainerConstPointer;
  typedef typename StatisticalModelMatrixContainerType::ConstIterator             StatisticalModelMatrixContainerConstIterator;

  itkSetConstObjectMacro( StatisticalModelContainer, StatisticalModelContainerType );
  itkGetConstObjectMacro( StatisticalModelContainer, StatisticalModelContainerType );

  void GetValueAndFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                              MeasureType & value,
                                              DerivativeType & derivative ) const;

  void GetModelValue( StatisticalModelConstPointer statisticalModel,
                             MeasureType & modelValue,
                             const TransformParametersType & parameters ) const;

  void GetModelFiniteDifferenceDerivative( StatisticalModelConstPointer statisticalModel,
                                                                     DerivativeType & modelDerivative,
                                                                     const TransformParametersType & parameters ) const;

  void TransformMesh( StatisticalModelMeshPointer fixedMesh, StatisticalModelMeshPointer movingMesh ) const;

  /** Initialize the Metric by making sure that all the components are
  *  present and plugged together correctly.
  */
  virtual void Initialize( void );

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & Derivative ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType & Value, DerivativeType & Derivative ) const;

protected:

  ActiveRegistrationModelShapeMetric();
  virtual ~ActiveRegistrationModelShapeMetric();

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  mutable StatisticalModelContainerConstPointer m_StatisticalModelContainer;

private:

  ActiveRegistrationModelShapeMetric( const Self & );    // purposely not implemented
  void operator=( const Self & ); // purposely not implemented

  StatisticalModelPathVectorType m_ShapeModelFilenames;
  StatisticalModelPathVectorType m_ShapeDirectories;
  StatisticalModelPathVectorType m_ReferenceFilenames;
  StatisticalModelParameterVectorType m_NoiseVariance;
  StatisticalModelParameterVectorType m_TotalVariance;

}; // end class PointSetPenalty

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkActiveRegistrationModelShapeMetric.hxx"
#endif

#endif // end #ifndef __itkActiveRegistrationModelPointDistributionShapeMetric_h__

