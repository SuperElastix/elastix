/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageToVectorContainerFilter_h
#define __ImageToVectorContainerFilter_h

#include "itkVectorContainerSource.h"


namespace itk
{

  /** \class ImageToVectorContainerFilter
   *
   * \brief Base class that takes in an image and pops out
   * a vector container.
   */

  template < class TInputImage, class TOutputVectorContainer >
  class ImageToVectorContainerFilter :
    public VectorContainerSource< TOutputVectorContainer >
  {
  public:

    /** Standard ITK-stuff. */
    typedef ImageToVectorContainerFilter  Self;
    typedef VectorContainerSource<
      TOutputVectorContainer >            Superclass;
    typedef SmartPointer<Self>            Pointer;
    typedef SmartPointer<const Self>      ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ImageToVectorContainerFilter, VectorContainerSource );

    /** Typedefs inherited from the superclass. */
    typedef typename Superclass::DataObjectPointer            DataObjectPointer;
    typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
    typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;

    /** Some Image related typedefs. */
    typedef TInputImage                             InputImageType;
    typedef typename InputImageType::Pointer        InputImagePointer;
    typedef typename InputImageType::ConstPointer   InputImageConstPointer;
    typedef typename InputImageType::RegionType     InputImageRegionType;
    typedef typename InputImageType::PixelType      InputImagePixelType;

    /** Create a valid output. */
    DataObject::Pointer MakeOutput(unsigned int idx);

    /** Set the input image of this process object.  */
    void SetInput( unsigned int idx, const InputImageType *input );

    /** Set the input image of this process object.  */
    void SetInput( const InputImageType *input );

    /** Get the input image of this process object.  */
    const InputImageType * GetInput( void );

    /** Get the input image of this process object.  */
    const InputImageType * GetInput( unsigned int idx );

    /** Get the output Mesh of this process object.  */
    OutputVectorContainerType * GetOutput( void );

    /** Prepare the output. */
    //virtual void GenerateOutputInformation( void );

  protected:

    /** The constructor. */
    ImageToVectorContainerFilter();
    /** The destructor. */
    virtual ~ImageToVectorContainerFilter() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

  private:

    /** The private constructor. */
    ImageToVectorContainerFilter( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );                // purposely not implemented

  }; // end class ImageToVectorContainerFilter


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToVectorContainerFilter.txx"
#endif

#endif // end #ifndef __ImageToVectorContainerFilter_h

