/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef _itkImageFileCastWriter_hxx
#define _itkImageFileCastWriter_hxx

#include "itkImageFileCastWriter.h"
#include "itkDataObject.h"
#include "itkObjectFactoryBase.h"
#include "itkImageIOFactory.h"
#include "itkCommand.h"
#include "vnl/vnl_vector.h"
#include "itkVectorImage.h"
#include "itkDefaultConvertPixelTraits.h"
#include "itkMetaImageIO.h"

namespace itk
{

//---------------------------------------------------------
template< class TInputImage >
ImageFileCastWriter< TInputImage >
::ImageFileCastWriter()
{
  this->m_Caster              = 0;
  this->m_OutputComponentType = this->GetDefaultOutputComponentType();
}


//---------------------------------------------------------
template< class TInputImage >
std::string
ImageFileCastWriter< TInputImage >
::GetDefaultOutputComponentType( void ) const
{
  /** Make a dummy imageIO object, which has some handy functions */
  MetaImageIO::Pointer dummyImageIO = MetaImageIO::New();

  /** Set the pixeltype. */
  typedef typename InputImageType::InternalPixelType ScalarType;
  //dummyImageIO->SetPixelTypeInfo(typeid(ScalarType));
  dummyImageIO->SetPixelTypeInfo( static_cast< const ScalarType * >( 0 ) );

  /** Get its description. */
  return dummyImageIO->GetComponentTypeAsString(
    dummyImageIO->GetComponentType() );
}


//---------------------------------------------------------
template< class TInputImage >
ImageFileCastWriter< TInputImage >
::~ImageFileCastWriter()
{
  this->m_Caster = 0;
}


//---------------------------------------------------------
template< class TInputImage >
void
ImageFileCastWriter< TInputImage >
::GenerateData( void )
{
  const InputImageType * input = this->GetInput();

  itkDebugMacro( << "Writing file: " << this->GetFileName() );

  // Make sure that the image is the right type and no more than
  // four components.
  typedef typename InputImageType::PixelType ScalarType;

  if( strcmp( input->GetNameOfClass(), "VectorImage" ) == 0 )
  {
    typedef typename InputImageType::InternalPixelType VectorImageScalarType;
    //this->GetImageIO()->SetPixelTypeInfo( typeid(VectorImageScalarType) );
    this->GetImageIO()->SetPixelTypeInfo( static_cast< const VectorImageScalarType * >( 0 ) );

    typedef typename InputImageType::AccessorFunctorType AccessorFunctorType;
    this->GetImageIO()->SetNumberOfComponents( AccessorFunctorType::GetVectorLength( input ) );
  }
  else
  {
    // Set the pixel and component type; the number of components.
    //this->GetImageIO()->SetPixelTypeInfo(typeid(ScalarType));
    this->GetImageIO()->SetPixelTypeInfo( static_cast< const ScalarType * >( 0 ) );
  }

  /** Setup the image IO for writing. */
  this->GetImageIO()->SetFileName( this->GetFileName() );

  /** Get the number of Components */
  unsigned int numberOfComponents = this->GetImageIO()->GetNumberOfComponents();

  /** Extract the data as a raw buffer pointer and possibly convert.
   * Converting is only possible if the number of components equals 1 */
  if(
    this->m_OutputComponentType !=
    this->GetImageIO()->GetComponentTypeAsString( this->GetImageIO()->GetComponentType() )
    && numberOfComponents == 1 )
  {
    void *             convertedDataBuffer = 0;
    const DataObject * inputAsDataObject
      = dynamic_cast< const DataObject * >( input );

    /** convert the scalar image to a scalar image with another componenttype
     * The imageIO's PixelType is also changed */
    if( this->m_OutputComponentType == "char" )
    {
      char dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy );
    }
    else if( this->m_OutputComponentType == "unsigned_char" )
    {
      unsigned char dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy );
    }
    else if( this->m_OutputComponentType == "short" )
    {
      short dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy    );
    }
    else if( this->m_OutputComponentType == "unsigned_short" )
    {
      unsigned short dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }
    else if( this->m_OutputComponentType == "int" )
    {
      int dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }
    else if( this->m_OutputComponentType == "unsigned_int" )
    {
      unsigned int dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }
    else if( this->m_OutputComponentType == "long" )
    {
      long dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }
    else if( this->m_OutputComponentType == "unsigned_long" )
    {
      unsigned long dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }
    else if( this->m_OutputComponentType == "float" )
    {
      float dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }
    else if( this->m_OutputComponentType == "double" )
    {
      double dummy;
      convertedDataBuffer = this->ConvertScalarImage( inputAsDataObject, dummy   );
    }

    /** Do the writing */
    this->GetImageIO()->Write( convertedDataBuffer );
    /** Release the caster's memory */
    this->m_Caster = 0;

  }
  else
  {
    /** No casting needed or possible, just write */
    const void * dataPtr = (const void *)input->GetBufferPointer();
    this->GetImageIO()->Write( dataPtr );
  }

}


} // end namespace itk

#endif
