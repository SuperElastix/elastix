/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __VectorContainerSource_h
#define __VectorContainerSource_h

#include "itkProcessObject.h"
#include "itkDataObjectDecorator.h"


namespace itk
{

  /** \class VectorContainerSource
   *
   * \brief A base class for creating an ImageToVectorContainerFilter.
   */

  template < class TOutputVectorContainer >
  class VectorContainerSource :
    public ProcessObject
  {
  public:

    /** Standard ITK-stuff. */
    typedef VectorContainerSource         Self;
    typedef ProcessObject                 Superclass;
    typedef SmartPointer<Self>            Pointer;
    typedef SmartPointer<const Self>      ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( VectorContainerSource, ProcessObject );

    /** Some convenient typedefs. */
    typedef typename Superclass::DataObjectPointer        DataObjectPointer;
    typedef TOutputVectorContainer                        OutputVectorContainerType;
    typedef typename OutputVectorContainerType::Pointer   OutputVectorContainerPointer;

    /** Get the vector container output of this process object. */
    OutputVectorContainerType * GetOutput( void );

    /** Get the vector container output of this process object. */
    OutputVectorContainerType * GetOutput( unsigned int idx );

    /** Graft the specified DataObject onto this ProcessObject's output. */
    virtual void GraftOutput( DataObject *output );

    /** Graft the specified DataObject onto this ProcessObject's output. */
    virtual void GraftNthOutput( unsigned int idx, DataObject *output );

    /** Make a DataObject of the correct type to used as the specified output. */
    virtual DataObjectPointer MakeOutput( unsigned int idx );

  protected:

    /** The constructor. */
    VectorContainerSource();
    /** The destructor. */
    virtual ~VectorContainerSource() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** GenerateInputRequestedRegion. */
    void GenerateInputRequestedRegion( void );

  private:

    /** The private constructor. */
    VectorContainerSource( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );        // purposely not implemented

    /** Member variables. */
    int m_GenerateDataRegion;
    int m_GenerateDataNumberOfRegions;

  }; // end class VectorContainerSource


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVectorContainerSource.txx"
#endif

#endif // end #ifndef __VectorContainerSource_h

