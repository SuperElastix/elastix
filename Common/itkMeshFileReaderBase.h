/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMeshFileReaderBase_h
#define __itkMeshFileReaderBase_h

#include "itkMeshSource.h"
#include "itkExceptionObject.h"
#include "itkMeshFileReader.h" // for MeshFileReaderException

namespace itk
{

/** \class MeshFileReaderBase
 *
 * \brief Base class for mesh readers
 *
 * A base class for classes that read a file containing
 * a mesh or a pointset.
 */

template< class TOutputMesh >
class MeshFileReaderBase : public MeshSource< TOutputMesh >
{
public:

  /** Standard class typedefs. */
  typedef MeshFileReaderBase         Self;
  typedef MeshSource< TOutputMesh >  Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MeshFileReaderBase, MeshSource );

  /** Some convenient typedefs. */
  typedef typename Superclass::DataObjectPointer DatabObjectPointer;
  typedef typename Superclass::OutputMeshType    OutputMeshType;
  typedef typename Superclass::OutputMeshPointer OutputMeshPointer;

  /** Set/Get the filename */
  itkGetStringMacro( FileName );
  itkSetStringMacro( FileName );

  /** Prepare the allocation of the output mesh during the first back
   * propagation of the pipeline.
   */
  virtual void GenerateOutputInformation( void );

  /** Give the reader a chance to indicate that it will produce more
   * output than it was requested to produce. MeshFileReader cannot
   * currently read a portion of a mesh, so the MeshFileReader must
   * enlarge the RequestedRegion to the size of the mesh on disk.
   */
  virtual void EnlargeOutputRequestedRegion( DataObject * output );

protected:

  MeshFileReaderBase();
  virtual ~MeshFileReaderBase(){}

  /** Test whether the given filename exist and it is readable,
   * this is intended to be called before attempting to use
   * subclasses for actually reading the file. If the file
   * doesn't exist or it is not readable, and exception with an
   * appropriate message will be thrown.
   */
  virtual void TestFileExistanceAndReadability( void );

  std::string m_FileName;

private:

  MeshFileReaderBase( const Self & );  // purposely not implemented
  void operator=( const Self & );      // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMeshFileReaderBase.hxx"
#endif

#endif
