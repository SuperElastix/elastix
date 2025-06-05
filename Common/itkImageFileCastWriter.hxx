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
#include "itkDeref.h"
#include "itkObjectFactoryBase.h"
#include "itkImageIOFactory.h"
#include "itkCommand.h"
#include <vnl/vnl_vector.h>
#include "itkVectorImage.h"
#include "itkDefaultConvertPixelTraits.h"
#include "itkMetaImageIO.h"
#include <iomanip>

namespace itk
{

//---------------------------------------------------------
template <typename TInputImage>
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
template <typename TInputImage>
void
ImageFileCastWriter<TInputImage>::GenerateData()
{
  const InputImageType & input = Deref(this->GetInput());

  itkDebugMacro("Writing file: " << this->GetFileName());

  ImageIOBase & imageIO = Deref(this->GetModifiableImageIO());

  // Make sure that the image is the right type and no more than
  // four components.
  using ScalarType = typename InputImageType::PixelType;

  if (strcmp(input.GetNameOfClass(), "VectorImage") == 0)
  {
    using VectorImageScalarType = typename InputImageType::InternalPixelType;
    imageIO.SetPixelTypeInfo(static_cast<const VectorImageScalarType *>(nullptr));

    using AccessorFunctorType = typename InputImageType::AccessorFunctorType;
    imageIO.SetNumberOfComponents(AccessorFunctorType::GetVectorLength(&input));
  }
  else
  {
    // Set the pixel and component type; the number of components.
    imageIO.SetPixelTypeInfo(static_cast<const ScalarType *>(nullptr));
  }

  /** Setup the image IO for writing. */
  imageIO.SetFileName(this->GetFileName());

  /** Get the number of Components */
  unsigned int numberOfComponents = imageIO.GetNumberOfComponents();

  /** Extract the data as a raw buffer pointer and possibly convert.
   * Converting is only possible if the number of components equals 1 */
  if (this->m_OutputComponentType != ImageIOBase::GetComponentTypeAsString(imageIO.GetComponentType()) &&
      numberOfComponents == 1)
  {
    const void * const convertedDataBuffer = [this, &input] {
      /** convert the scalar image to a scalar image with another componenttype
       * The imageIO's PixelType is also changed */
      if (this->m_OutputComponentType == "char")
      {
        return this->ConvertScalarImage<char>(input);
      }
      if (this->m_OutputComponentType == "unsigned_char")
      {
        return this->ConvertScalarImage<unsigned char>(input);
      }
      if (this->m_OutputComponentType == "short")
      {
        return this->ConvertScalarImage<short>(input);
      }
      if (this->m_OutputComponentType == "unsigned_short")
      {
        return this->ConvertScalarImage<unsigned short>(input);
      }
      if (this->m_OutputComponentType == "int")
      {
        return this->ConvertScalarImage<int>(input);
      }
      if (this->m_OutputComponentType == "unsigned_int")
      {
        return this->ConvertScalarImage<unsigned int>(input);
      }
      if (this->m_OutputComponentType == "long")
      {
        return this->ConvertScalarImage<long>(input);
      }
      if (this->m_OutputComponentType == "unsigned_long")
      {
        return this->ConvertScalarImage<unsigned long>(input);
      }
      if (this->m_OutputComponentType == "float")
      {
        return this->ConvertScalarImage<float>(input);
      }
      if (this->m_OutputComponentType == "double")
      {
        return this->ConvertScalarImage<double>(input);
      }
      itkExceptionMacro("Unable to convert the input image. An unknown or unsupported component type was specified: "
                        << std::quoted(m_OutputComponentType) << ".");
    }();

    /** Do the writing */
    imageIO.Write(convertedDataBuffer);
    /** Release the caster's memory */
    this->m_Caster = nullptr;
  }
  else
  {
    /** No casting needed or possible, just write */
    const void * dataPtr = input.GetBufferPointer();
    imageIO.Write(dataPtr);
  }
}


} // end namespace itk

#endif
