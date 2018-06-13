/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkTransformixInputPointFileReader_hxx
#define __itkTransformixInputPointFileReader_hxx

#include "itkTransformixInputPointFileReader.h"

namespace itk
{

/**
 * **************** Constructor ***************
 */

template< class TOutputMesh >
TransformixInputPointFileReader< TOutputMesh >
::TransformixInputPointFileReader()
{
  this->m_NumberOfPoints   = 0;
  this->m_PointsAreIndices = false;
} // end constructor


/**
 * **************** Destructor ***************
 */

template< class TOutputMesh >
TransformixInputPointFileReader< TOutputMesh >
::~TransformixInputPointFileReader()
{
  if( this->m_Reader.is_open() )
  {
    this->m_Reader.close();
  }
} // end constructor


/**
 * ***************GenerateOutputInformation ***********
 */

template< class TOutputMesh >
void
TransformixInputPointFileReader< TOutputMesh >
::GenerateOutputInformation( void )
{
  this->Superclass::GenerateOutputInformation();

  /** The superclass tests already if it's a valid file; so just open it and
  * assume it goes alright */
  if( this->m_Reader.is_open() )
  {
    this->m_Reader.close();
  }
  this->m_Reader.open( this->m_FileName.c_str() );

  /** Read the first entry */
  std::string indexOrPoint;
  this->m_Reader >> indexOrPoint;

  /** Set the IsIndex bool and the number of points.*/
  if( indexOrPoint == "point" )
  {
    /** Input points are specified in world coordinates. */
    this->m_PointsAreIndices = false;
    this->m_Reader >> this->m_NumberOfPoints;
  }
  else if( indexOrPoint == "index" )
  {
    /** Input points are specified as image indices. */
    this->m_PointsAreIndices = true;
    this->m_Reader >> this->m_NumberOfPoints;
  }
  else
  {
    /** Input points are assumed to be specified as image indices. */
    this->m_PointsAreIndices = true;
    this->m_NumberOfPoints   = atoi( indexOrPoint.c_str() );
  }

  /** Leave the file open for the generate data method */

} // end GenerateOutputInformation()


/**
 * ***************GenerateData ***********
 */

template< class TOutputMesh >
void
TransformixInputPointFileReader< TOutputMesh >
::GenerateData( void )
{
  typedef typename OutputMeshType::PointsContainer PointsContainerType;
  typedef typename PointsContainerType::Pointer    PointsContainerPointer;
  typedef typename OutputMeshType::PointType       PointType;
  const unsigned int dimension = OutputMeshType::PointDimension;

  OutputMeshPointer      output = this->GetOutput();
  PointsContainerPointer points = PointsContainerType::New();

  /** Read the file */
  if( this->m_Reader.is_open() )
  {
    for( unsigned int i = 0; i < this->m_NumberOfPoints; ++i )
    {
      // read point from textfile
      PointType point;
      for( unsigned int j = 0; j < dimension; j++ )
      {
        if( !this->m_Reader.eof() )
        {
          this->m_Reader >> point[ j ];
        }
        else
        {
          std::ostringstream msg;
          msg << "The file is not large enough. "
              << std::endl << "Filename: " << this->m_FileName
              << std::endl;
          MeshFileReaderException e( __FILE__, __LINE__, msg.str().c_str(), ITK_LOCATION );
          throw e;
          return;
        }
      }
      points->push_back( point );
    }
  }
  else
  {
    std::ostringstream msg;
    msg << "The file has unexpectedly been closed. "
        << std::endl << "Filename: " << this->m_FileName
        << std::endl;
    MeshFileReaderException e( __FILE__, __LINE__, msg.str().c_str(), ITK_LOCATION );
    throw e;
    return;
  }

  /** set in output */
  output->Initialize();
  output->SetPoints( points );

  /** Close the reader */
  this->m_Reader.close();

  /** This indicates that the current BufferedRegion is equal to the
   * requested region. This action prevents useless re-executions of
   * the pipeline.
   * (I copied this from the BinaryMaskToNarrowBandPointSetFilter) */
  output->SetBufferedRegion( output->GetRequestedRegion() );

} // end GenerateData()


} // end namespace itk

#endif
