/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkImageFileCastWriter_h
#define __itkImageFileCastWriter_h

#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkExceptionObject.h"
#include "itkSize.h"
#include "itkImageIORegion.h"
#include "itkCastImageFilter.h"

namespace itk
{

/** \class ImageFileCastWriter
 * \brief Casts pixel type and writes image data.
 *
 * This filter saves an image and casts the data on the fly,
 * if necessary. This is useful in some cases, to avoid the use of
 * a itk::CastImageFilter (to save memory for example).
 *
 */
template< class TInputImage >
class ImageFileCastWriter : public ImageFileWriter< TInputImage >
{
public:

  /** Standard class typedefs. */
  typedef ImageFileCastWriter            Self;
  typedef ImageFileWriter< TInputImage > Superclass;
  typedef SmartPointer< Self >           Pointer;
  typedef SmartPointer< const Self >     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageFileCastWriter, ImageFileWriter );

  /** Some convenient typedefs. */
  typedef typename Superclass::InputImageType       InputImageType;
  typedef typename Superclass::InputImagePointer    InputImagePointer;
  typedef typename Superclass::InputImageRegionType InputImageRegionType;
  typedef typename Superclass::InputImagePixelType  InputImagePixelType;

  itkStaticConstMacro( InputImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Set the component type for writing to disk; default: the same as
   * the InputImagePixelType::ComponentType. This setting is ignored when
   * the inputImagePixelType is not a scalar*/
  itkSetStringMacro( OutputComponentType );
  itkGetStringMacro( OutputComponentType );

  /** Determine the default outputcomponentType */
  std::string GetDefaultOutputComponentType( void ) const;

protected:

  ImageFileCastWriter();
  ~ImageFileCastWriter();

  /** Does the real work. */
  void GenerateData( void );

  /** Templated function that casts the input image and returns a
   * a pointer to the PixelBuffer. Assumes scalar singlecomponent images
   * The buffer data is valid until this->m_Caster is destroyed or assigned
   * a new caster. The ImageIO's PixelType is also adapted by this function */
  template< class OutputComponentType >
  void * ConvertScalarImage( const DataObject * inputImage,
    const OutputComponentType & itkNotUsed( dummy ) )
  {
    typedef Image< OutputComponentType, InputImageDimension >      DiskImageType;
    typedef typename PixelTraits< InputImagePixelType >::ValueType InputImageComponentType;
    typedef Image< InputImageComponentType, InputImageDimension >  ScalarInputImageType;
    typedef CastImageFilter< ScalarInputImageType, DiskImageType > CasterType;

    /** Reconfigure the imageIO */
    //this->GetImageIO()->SetPixelTypeInfo( typeid(OutputComponentType) );
    this->GetImageIO()->SetPixelTypeInfo( static_cast< const OutputComponentType * >( 0 ) );

    /** cast the input image */
    typename CasterType::Pointer caster                    = CasterType::New();
    this->m_Caster                                         = caster;
    typename ScalarInputImageType::Pointer localInputImage = ScalarInputImageType::New();
    localInputImage->Graft( inputImage );
    caster->SetInput( localInputImage );
    caster->Update();

    /** return the pixel buffer of the casted image */
    OutputComponentType * pixelBuffer     = caster->GetOutput()->GetBufferPointer();
    void *                convertedBuffer = static_cast< void * >( pixelBuffer );
    return convertedBuffer;
  }


  ProcessObject::Pointer m_Caster;

private:

  ImageFileCastWriter( const Self & ); // purposely not implemented
  void operator=( const Self & );      // purposely not implemented

  std::string m_OutputComponentType;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageFileCastWriter.hxx"
#endif

#endif // __itkImageFileCastWriter_h
