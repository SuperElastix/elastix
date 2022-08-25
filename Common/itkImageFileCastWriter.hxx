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
#include <vnl/vnl_vector.h>
#include "itkVectorImage.h"
#include "itkDefaultConvertPixelTraits.h"
#include "itkMetaImageIO.h"

namespace itk
{

//---------------------------------------------------------
template <class TInputImage>
std::string
ImageFileCastWriter<TInputImage>::GetDefaultOutputComponentType()
{
  /** Make a dummy imageIO object, which has some handy functions */
  const auto dummyImageIO = MetaImageIO::New();

  /** Set the pixeltype. */
  using ScalarType = typename InputImageType::InternalPixelType;

  dummyImageIO->SetPixelTypeInfo(static_cast<const ScalarType *>(nullptr));

  /** Get its description. */
  return ImageIOBase::GetComponentTypeAsString(dummyImageIO->GetComponentType());
}


//---------------------------------------------------------
template <class TInputImage>
void
ImageFileCastWriter<TInputImage>::GenerateData()
{
  const InputImageType * input = this->GetInput();

  itkDebugMacro(<< "Writing file: " << this->GetFileName());

  // Make sure that the image is the right type and no more than
  // four components.
  using ScalarType = typename InputImageType::PixelType;

  if (strcmp(input->GetNameOfClass(), "VectorImage") == 0)
  {
    using VectorImageScalarType = typename InputImageType::InternalPixelType;
    // this->GetImageIO()->SetPixelTypeInfo( typeid(VectorImageScalarType) );
    this->GetModifiableImageIO()->SetPixelTypeInfo(static_cast<const VectorImageScalarType *>(nullptr));

    using AccessorFunctorType = typename InputImageType::AccessorFunctorType;
    this->GetModifiableImageIO()->SetNumberOfComponents(AccessorFunctorType::GetVectorLength(input));
  }
  else
  {
    // Set the pixel and component type; the number of components.
    // this->GetImageIO()->SetPixelTypeInfo(typeid(ScalarType));
    this->GetModifiableImageIO()->SetPixelTypeInfo(static_cast<const ScalarType *>(nullptr));
  }

  /** Setup the image IO for writing. */
  this->GetModifiableImageIO()->SetFileName(this->GetFileName());

  /** Get the number of Components */
  unsigned int numberOfComponents = this->GetImageIO()->GetNumberOfComponents();

  /** Extract the data as a raw buffer pointer and possibly convert.
   * Converting is only possible if the number of components equals 1 */
  if (this->m_OutputComponentType !=
        this->GetImageIO()->GetComponentTypeAsString(this->GetImageIO()->GetComponentType()) &&
      numberOfComponents == 1)
  {
    void *             convertedDataBuffer = nullptr;
    const DataObject * inputAsDataObject = dynamic_cast<const DataObject *>(input);

    /** convert the scalar image to a scalar image with another componenttype
     * The imageIO's PixelType is also changed */
    if (this->m_OutputComponentType == "char")
    {
      convertedDataBuffer = this->ConvertScalarImage<char>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "unsigned_char")
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned char>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "short")
    {
      convertedDataBuffer = this->ConvertScalarImage<short>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "unsigned_short")
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned short>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "int")
    {
      convertedDataBuffer = this->ConvertScalarImage<int>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "unsigned_int")
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned int>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "long")
    {
      convertedDataBuffer = this->ConvertScalarImage<long>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "unsigned_long")
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned long>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "float")
    {
      convertedDataBuffer = this->ConvertScalarImage<float>(inputAsDataObject);
    }
    else if (this->m_OutputComponentType == "double")
    {
      convertedDataBuffer = this->ConvertScalarImage<double>(inputAsDataObject);
    }

    /** Do the writing */
    this->GetModifiableImageIO()->Write(convertedDataBuffer);
    /** Release the caster's memory */
    this->m_Caster = nullptr;
  }
  else
  {
    /** No casting needed or possible, just write */
    const void * dataPtr = input->GetBufferPointer();
    this->GetModifiableImageIO()->Write(dataPtr);
  }
}


} // end namespace itk

#endif
