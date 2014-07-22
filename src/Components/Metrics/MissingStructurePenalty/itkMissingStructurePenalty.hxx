/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMissingStructurePenalty_hxx
#define __itkMissingStructurePenalty_hxx

#include "itkMissingStructurePenalty.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >
::MissingVolumeMeshPenalty()
{
  this->m_MappedMeshContainer = MappedMeshContainerType::New();
} // end Constructor


/**
 * ******************* Destructor *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet  >
::~MissingVolumeMeshPenalty()
{} // end Destructor


/**
 * *********************** Initialize *****************************
 */

template< class TFixedPointSet, class TMovingPointSet  >
void
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the initialize of the superclass. */
  //this->Superclass::Initialize();

  if( !this->m_Transform )
  {
    itkExceptionMacro( << "Transform is not present" );
  }

  if( !this->m_FixedMeshContainer )
  {
    itkExceptionMacro( << "FixedMeshContainer is not present" );
  }

  const FixedMeshContainerType::ElementIdentifier numberOfMeshes = this->m_FixedMeshContainer->Size();
  this->m_MappedMeshContainer->Reserve( numberOfMeshes );

  for( FixedMeshContainerType::ElementIdentifier meshId = 0; meshId < numberOfMeshes; ++meshId )
  {
    FixedMeshConstPointer           fixedMesh      = this->m_FixedMeshContainer->ElementAt( meshId );
    MeshPointsContainerConstPointer fixedPoints    = fixedMesh->GetPoints();
    const unsigned int              numberOfPoints = fixedPoints->Size();

    MeshPointsContainerType::Pointer mappedPoints =  MeshPointsContainerType::New();
    mappedPoints->Reserve( numberOfPoints );

    FixedMeshType::Pointer mappedMesh = FixedMeshType::New();
    mappedMesh->SetPoints( mappedPoints );

    // mappedMesh was constructed with a Cellscontainer and CellDatacontainer of size 0.
    // We use a null pointer to set them to undefined, which is also the default behavior of the MeshReader.
    // "Write result mesh" checks the null pointer and writes a mesh with the remaining data filled in from the fixed mesh.
    mappedMesh->SetPointData( NULL );
    mappedMesh->SetCells( NULL );
    mappedMesh->SetCellData( NULL );

    this->m_MappedMeshContainer->SetElement( meshId, mappedMesh );

  }
} // end Initialize()


/**
 * ******************* GetValue *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
typename MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >::MeasureType
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >
::GetValue( const TransformParametersType & parameters ) const
{
  /** Sanity checks. */
  FixedMeshContainerConstPointer fixedMeshContainer = this->GetFixedMeshContainer();
  if( !fixedMeshContainer )
  {
    itkExceptionMacro( << "FixedMeshContainer mesh has not been assigned" );
  }

  /** Initialize some variables */
  MeasureType value = NumericTraits< MeasureType >::Zero;

  OutputPointType fixedPoint;
  /** Get the current corresponding points. */

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  DerivativeType dummyDerivative;
  this->GetValueAndDerivative( parameters, value, dummyDerivative );

  return value;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >
::GetDerivative( const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
  * the metric value is available. It does not cost anything to calculate
  * the metric value now. Therefore, we have chosen to only implement the
  * GetValueAndDerivative(), supplying it with a dummy value variable.
  */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >
::GetValueAndDerivative( const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Sanity checks. */
  FixedMeshContainerConstPointer fixedMeshContainer = this->GetFixedMeshContainer();
  if( !fixedMeshContainer )
  {
    itkExceptionMacro( << "FixedMeshContainer mesh has not been assigned" );
  }

  /** Initialize some variables */
  value = NumericTraits< MeasureType >::Zero;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  NonZeroJacobianIndicesType nzji( this->m_Transform->GetNumberOfNonZeroJacobianIndices() );
  TransformJacobianType      jacobian;

  const FixedMeshContainerType::ElementIdentifier numberOfMeshes = this->m_FixedMeshContainer->Size();

  FixedMeshType::PointType zeroPoint =  FixedMeshType::PointType::Point();
  zeroPoint.Fill( 0.0 );

  MeshPointsContainerType::Pointer pointCentroids = FixedMeshType::PointsContainer::New();
  pointCentroids->resize( numberOfMeshes, FixedMeshType::PointType::Point( zeroPoint ) );

  for( FixedMeshContainerType::ElementIdentifier meshId = 0; meshId < numberOfMeshes; ++meshId ) // loop over all meshes in container
  {
    const FixedMeshConstPointer           fixedMesh      = fixedMeshContainer->ElementAt( meshId );
    const MeshPointsContainerConstPointer fixedPoints    = fixedMesh->GetPoints();
    const unsigned int                    numberOfPoints = fixedPoints->Size();

    const FixedMeshPointer           mappedMesh   = this->m_MappedMeshContainer->ElementAt( meshId );
    const MeshPointsContainerPointer mappedPoints = mappedMesh->GetPoints();

    FixedMeshType::PointType &              pointCentroid = pointCentroids->ElementAt( meshId );
    FixedMeshType::PointsContainer::Pointer derivPoints   = FixedMeshType::PointsContainer::New();

    derivPoints->resize( numberOfPoints, FixedMeshType::PointType::Point( zeroPoint ) );

    MeshPointsContainerConstIteratorType fixedPointIt  = fixedPoints->Begin();
    MeshPointsContainerIteratorType      mappedPointIt = mappedPoints->Begin();
    MeshPointsContainerConstIteratorType fixedPointEnd = fixedPoints->End();

    for( fixedPointIt; fixedPointIt != fixedPointEnd; ++fixedPointIt, ++mappedPointIt )
    {
      const OutputPointType mappedPoint = this->m_Transform->TransformPoint( fixedPointIt->Value() );
      mappedPointIt.Value()         = mappedPoint;
      pointCentroid.GetVnlVector() += mappedPoint.GetVnlVector();
    }
    pointCentroid.GetVnlVector() /= numberOfPoints;

    FixedMeshType::CellsContainerConstIterator cellBegin = fixedMesh->GetCells()->Begin();
    FixedMeshType::CellsContainerConstIterator cellEnd   = fixedMesh->GetCells()->End();

    CellInterfaceType::PointIdIterator beginpointer;
    float                              sumSignedVolume = 0.0;
    float                              sumAbsVolume    = 0.0;

    const int eps = 0.00001;

    for( cellBegin; cellBegin != cellEnd; ++cellBegin )
    {
      beginpointer = cellBegin->Value()->PointIdsBegin();
      float signedVolume;  // = vnl_determinant(fullMatrix.GetVnlMatrix());

      //const VectorType::const_pointer p1,p2,p3,p4;
      switch( FixedPointSetDimension )
      {
        case 2:
        {
          const FixedMeshType::PointIdentifier p1Id = *beginpointer;
          ++beginpointer;
          const VectorType                     p1   = mappedPoints->GetElement( p1Id ) - pointCentroid;
          const FixedMeshType::PointIdentifier p2Id = *beginpointer;
          ++beginpointer;
          const VectorType p2 = mappedPoints->GetElement( p2Id ) - pointCentroid;

          signedVolume = vnl_determinant( p1.GetDataPointer(), p2.GetDataPointer() );

          const int sign = ( signedVolume > eps ) - ( signedVolume < -eps );
          if( sign != 0 )
          {
            derivPoints->at( p1Id )[ 0 ] += sign * p2[ 1 ];
            derivPoints->at( p1Id )[ 1 ] -= sign * p2[ 0 ];
            derivPoints->at( p2Id )[ 0 ] -= sign * p1[ 1 ];
            derivPoints->at( p2Id )[ 1 ] += sign * p1[ 0 ];
          }

        }
        break;
        case 3:
        {
          const FixedMeshType::PointIdentifier p1Id = *beginpointer;
          ++beginpointer;
          const VectorType                     p1   = mappedPoints->GetElement( p1Id ) - pointCentroid;
          const FixedMeshType::PointIdentifier p2Id = *beginpointer;
          ++beginpointer;
          const VectorType                     p2   = mappedPoints->GetElement( p2Id ) - pointCentroid;
          const FixedMeshType::PointIdentifier p3Id = *beginpointer;
          ++beginpointer;
          const VectorType p3 = mappedPoints->GetElement( p3Id ) - pointCentroid;

          signedVolume = vnl_determinant( p1.GetDataPointer(), p2.GetDataPointer(), p3.GetDataPointer() );

          const int sign = ( ( signedVolume > eps ) - ( signedVolume < -eps ) );

          if( sign != 0 )
          {
            derivPoints->at( p1Id )[ 0 ] += sign * ( p2[ 1 ] * p3[ 2 ] - p2[ 2 ] * p3[ 1 ] );
            derivPoints->at( p1Id )[ 1 ] += sign * ( p2[ 2 ] * p3[ 0 ] - p2[ 0 ] * p3[ 2 ] );
            derivPoints->at( p1Id )[ 2 ] += sign * ( p2[ 0 ] * p3[ 1 ] - p2[ 1 ] * p3[ 0 ] );

            derivPoints->at( p2Id )[ 0 ] += sign * ( p1[ 2 ] * p3[ 1 ] - p1[ 1 ] * p3[ 2 ] );
            derivPoints->at( p2Id )[ 1 ] += sign * ( p1[ 0 ] * p3[ 2 ] - p1[ 2 ] * p3[ 0 ] );
            derivPoints->at( p2Id )[ 2 ] += sign * ( p1[ 1 ] * p3[ 0 ] - p1[ 0 ] * p3[ 1 ] );

            derivPoints->at( p3Id )[ 0 ] += sign * ( p1[ 1 ] * p2[ 2 ] - p1[ 2 ] * p2[ 1 ] );
            derivPoints->at( p3Id )[ 1 ] += sign * ( p1[ 2 ] * p2[ 0 ] - p1[ 0 ] * p2[ 2 ] );
            derivPoints->at( p3Id )[ 2 ] += sign * ( p1[ 0 ] * p2[ 1 ] - p1[ 1 ] * p2[ 0 ] );

          }
        }

        break;
        case 4:
        {
          const VectorType::const_pointer p1 = mappedPoints->GetElement( *beginpointer++ ).GetDataPointer();
          const VectorType::const_pointer p2 = mappedPoints->GetElement( *beginpointer++ ).GetDataPointer();
          const VectorType::const_pointer p3 = mappedPoints->GetElement( *beginpointer++ ).GetDataPointer();
          const VectorType::const_pointer p4 = mappedPoints->GetElement( *beginpointer++ ).GetDataPointer();
          signedVolume = vnl_determinant( p1, p2, p3, p4 );
        }
        break;
        default:
          std::cout << "no dimensions higher than 4"  << std::endl;
      }

      sumSignedVolume +=  signedVolume;
      sumAbsVolume    += abs( signedVolume );
    }

    /** Create iterators. */
    fixedPointIt = fixedPoints->Begin();
    unsigned int pointIndex;
    /** Loop over points. */
    for( fixedPointIt, pointIndex = 0; fixedPointIt != fixedPointEnd; ++fixedPointIt, ++pointIndex )
    {
      /** Get the TransformJacobian dT/dmu. */
      this->m_Transform->GetJacobian( fixedPointIt.Value(), jacobian, nzji );
      if( nzji.size() == this->GetNumberOfParameters() )
      {
        /** Loop over all Jacobians. */
        derivative += derivPoints->at( pointIndex ).GetVnlVector() * jacobian; //* sumAbsVolumeEps;
      }
      else
      {
        /** Only pick the nonzero Jacobians. */
        for( unsigned int i = 0; i < nzji.size(); ++i )
        {
          const unsigned int index  = nzji[ i ];
          VnlVectorType      column = jacobian.get_column( i );
          derivative[ index ] += dot_product( derivPoints->at( pointIndex ).GetVnlVector(), column ); // *sumAbsVolumeEps;
        }
      }

    }   // end loop over all corresponding points

    /** Check if enough samples were valid. */

    /** Copy the measure to value. */
    value += sumAbsVolume;

  } // end loop over all meshes in container
} // end GetValueAndDerivative()


/**
 * ******************* SubVector *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
MissingVolumeMeshPenalty< TFixedPointSet, TMovingPointSet >
::SubVector( const VectorType & fullVector,  SubVectorType & subVector, const unsigned int leaveOutIndex )  const
{
  //SubVectorType subVector = SubVectorType::Vector();

  VectorType::ConstIterator    fullVectorIt  = fullVector.Begin();
  VectorType::ConstIterator    fullVectorEnd = fullVector.End();
  SubVectorType::Iterator      subVectorIt   = subVector.Begin();
  SubVectorType::ConstIterator subVectorEnd  = subVector.End();
  unsigned int                 fullIndex     = 0;
  for( subVectorIt, fullIndex; subVectorIt != subVectorEnd; ++fullVectorIt, ++subVectorIt, ++fullIndex )
  {
    if( fullIndex == leaveOutIndex )
    {
      ++fullVectorIt;
    }
    *subVectorIt = *fullVectorIt;
  }

} // end SubVector()


} // end itk namespace

#endif // end #ifndef __itkMissingStructurePenalty_hxx

