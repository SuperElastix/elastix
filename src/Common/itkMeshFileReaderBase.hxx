/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMeshFileReaderBase_hxx
#define __itkMeshFileReaderBase_hxx

#include "itkMeshFileReaderBase.h"

#include <itksys/SystemTools.hxx>
#include <fstream>

namespace itk
{

/**
 * **************** Constructor ***************
 */

template <class TOutputMesh>
MeshFileReaderBase<TOutputMesh>
::MeshFileReaderBase()
{
  this->m_FileName = "";
} // end constructor


/**
 * ***************GenerateOutputInformation ***********
 */

template <class TOutputMesh>
void
MeshFileReaderBase<TOutputMesh>
::GenerateOutputInformation( void )
{
  OutputMeshPointer output = this->GetOutput();

  itkDebugMacro( << "Reading file for GenerateOutputInformation(): " << this->m_FileName );

  /** Check to see if we can read the file given the name or prefix */
  if ( this->m_FileName == "" )
  {
    throw MeshFileReaderException( __FILE__, __LINE__,
      "FileName must be specified", ITK_LOCATION );
  }

  /** Test if the file exist and if it can be open.
   * and exception will be thrown otherwise. */
  this->TestFileExistanceAndReadability();

  //Copy MetaDataDictionary from instantiated reader to output mesh?
  //output->SetMetaDataDictionary(m_ImageIO->GetMetaDataDictionary());
  //this->SetMetaDataDictionary(m_ImageIO->GetMetaDataDictionary());

  // This makes not really sense i think.
  //MeshRegionType region;
  // region = ?
  //output->SetLargestPossibleRegion(region);

}   // end GenerateOutputInformation()


/**
 * *************TestFileExistanceAndReadability ***********
 */

template <class TOutputMesh>
void
MeshFileReaderBase<TOutputMesh>
::TestFileExistanceAndReadability( void )
{
  // Test if the file exists.
  if( ! itksys::SystemTools::FileExists( this->m_FileName.c_str() ) )
  {
    MeshFileReaderException e( __FILE__, __LINE__ );
    std::ostringstream msg;
    msg <<"The file doesn't exists. "
      << std::endl << "Filename = " << this->m_FileName
      << std::endl;
    e.SetDescription( msg.str().c_str() );
    throw e;
    return;
  }

  // Test if the file can be open for reading access.
  std::ifstream readTester;
  readTester.open( this->m_FileName.c_str() );
  if( readTester.fail() )
  {
    readTester.close();
    std::ostringstream msg;
    msg <<"The file couldn't be opened for reading. "
      << std::endl << "Filename: " << this->m_FileName
      << std::endl;
    MeshFileReaderException e( __FILE__, __LINE__,
      msg.str().c_str(), ITK_LOCATION );
    throw e;
    return;

  }
  readTester.close();

} // end TestFileExistanceAndReadability()


/**
 * **************EnlargeOutputRequestedRegion***********
 */

template <class TOutputMesh>
void
MeshFileReaderBase<TOutputMesh>
::EnlargeOutputRequestedRegion( DataObject *output )
{
  OutputMeshPointer out = dynamic_cast<OutputMeshType *>( output );

  if( out )
  {
    out->SetRequestedRegionToLargestPossibleRegion();
  }
  else
  {
    throw  MeshFileReaderException( __FILE__, __LINE__,
      "Invalid output object type" );
  }

} // end EnlargeOutputRequestedRegion()


} // end namespace itk


#endif
