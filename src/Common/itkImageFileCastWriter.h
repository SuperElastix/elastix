/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
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
 * \brief Writes image data to a single file.
 *
 * ImageFileCastWriter writes its input data to a single output file.
 * ImageFileCastWriter interfaces with an ImageIO class to write out the
 * data. If you wish to write data into a series of files (e.g., a
 * slice per file) use ImageSeriesWriter.
 *
 * A pluggable factory pattern is used that allows different kinds of writers
 * to be registered (even at run time) without having to modify the
 * code in this class. You can either manually instantiate the ImageIO
 * object and associate it with the ImageFileCastWriter, or let the class
 * figure it out from the extension. Normally just setting the filename
 * with a suitable suffix (".png", ".jpg", etc) and setting the input 
 * to the writer is enough to get the writer to work properly.
 *
 * \sa ImageSeriesReader
 * \sa ImageIOBase
 *
 * \ingroup IOFilters 
 */
template <class TInputImage >
  class ImageFileCastWriter : public ImageFileWriter<TInputImage>
{
public:
  /** Standard class typedefs. */
  typedef ImageFileCastWriter                   Self;
  typedef ImageFileWriter<TInputImage>       Superclass;
  typedef SmartPointer<Self>                 Pointer;
  typedef SmartPointer<const Self>           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageFileCastWriter,ImageFileWriter);

  /** Some convenient typedefs. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageRegionType   InputImageRegionType;  
  typedef typename Superclass::InputImagePixelType    InputImagePixelType; 

  itkStaticConstMacro( InputImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Set the component type for writing to disk; default: the same as 
   * the InputImagePixelType::ComponentType. This setting is ignored when
   * the inputImagePixelType is not a scalar*/
  itkSetStringMacro(OutputComponentType);
  itkGetStringMacro(OutputComponentType);
  
  /** Determine the default outputcomponentType */
  std::string GetDefaultOutputComponentType(void) const;

protected:
  ImageFileCastWriter();
  ~ImageFileCastWriter();
 
  /** Does the real work. */
  void GenerateData(void);

  /** Templated function that casts the input image and returns a
   * a pointer to the PixelBuffer. Assumes scalar singlecomponent images
   * The buffer data is valid until this->m_Caster is destroyed or assigned
   * a new caster. The ImageIO's PixelType is also adapted by this function */
  template < class OutputComponentType >
    void * ConvertScalarImage( const DataObject * inputImage, const OutputComponentType & dummy )
  {
    typedef Image< OutputComponentType, InputImageDimension>      DiskImageType;
    typedef typename PixelTraits<InputImagePixelType>::ValueType  InputImageComponentType;
    typedef Image<InputImageComponentType, InputImageDimension>   ScalarInputImageType;
    typedef CastImageFilter< ScalarInputImageType, DiskImageType> CasterType;
    
    /** Reconfigure the imageIO */
    this->GetImageIO()->SetPixelTypeInfo( typeid(OutputComponentType) );

    /** cast the input image */
    typename CasterType::Pointer caster = CasterType::New();
    this->m_Caster = caster;
    caster->SetInput( dynamic_cast<const ScalarInputImageType *>(inputImage) );
    caster->Update();

    /** return the pixel buffer of the casted image */
    OutputComponentType * pixelBuffer = caster->GetOutput()->GetBufferPointer();
    void * convertedBuffer = static_cast<void *>(pixelBuffer); 
    return convertedBuffer;
  }

  ProcessObject::Pointer m_Caster;
    
private:
  ImageFileCastWriter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::string m_OutputComponentType;
};

  
} // end namespace itk
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageFileCastWriter.txx"
#endif

#endif // __itkImageFileCastWriter_h
  
