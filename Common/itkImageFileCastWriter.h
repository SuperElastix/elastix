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
#ifndef itkImageFileCastWriter_h
#define itkImageFileCastWriter_h

#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkMacro.h"
#include "itkSize.h"
#include "itkImageIORegion.h"
#include "itkCastImageFilter.h"
#include "elxDefaultConstruct.h"

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
template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageFileCastWriter : public ImageFileWriter<TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageFileCastWriter);

  /** Standard class typedefs. */
  using Self = ImageFileCastWriter;
  using Superclass = ImageFileWriter<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageFileCastWriter, ImageFileWriter);

  /** Some convenient typedefs. */
  using typename Superclass::InputImageType;
  using typename Superclass::InputImagePointer;
  using typename Superclass::InputImageRegionType;
  using typename Superclass::InputImagePixelType;

  itkStaticConstMacro(InputImageDimension, unsigned int, InputImageType::ImageDimension);

  /** Set the component type for writing to disk; default: the same as
   * the InputImagePixelType::ComponentType. This setting is ignored when
   * the inputImagePixelType is not a scalar*/
  itkSetStringMacro(OutputComponentType);
  itkGetStringMacro(OutputComponentType);

protected:
  ImageFileCastWriter() = default;
  ~ImageFileCastWriter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

private:
  /** Determine the default outputcomponentType */
  static std::string
  GetDefaultOutputComponentType();

  /** Templated function that casts the input image and returns a
   * a pointer to the PixelBuffer. Assumes scalar singlecomponent images
   * The buffer data is valid until this->m_Caster is destroyed or assigned
   * a new caster. The ImageIO's PixelType is also adapted by this function */
  template <class OutputComponentType>
  void *
  ConvertScalarImage(const DataObject * inputImage)
  {
    using DiskImageType = Image<OutputComponentType, InputImageDimension>;
    using InputImageComponentType = typename PixelTraits<InputImagePixelType>::ValueType;
    using ScalarInputImageType = Image<InputImageComponentType, InputImageDimension>;

    /** Reconfigure the imageIO */
    this->GetModifiableImageIO()->SetPixelTypeInfo(static_cast<const OutputComponentType *>(nullptr));

    /** cast the input image */
    const auto caster = CastImageFilter<ScalarInputImageType, DiskImageType>::New();
    this->m_Caster = caster;
    const auto localInputImage = ScalarInputImageType::New();

    localInputImage->Graft(static_cast<const ScalarInputImageType *>(inputImage));

    caster->SetInput(localInputImage);
    caster->Update();

    /** return the pixel buffer of the casted image */
    return caster->GetOutput()->GetBufferPointer();
  }


  ProcessObject::Pointer m_Caster{ nullptr };

  std::string m_OutputComponentType{ Self::GetDefaultOutputComponentType() };
};

/** Convenience function for writing a casted image. */
template <typename TImage>
void
WriteCastedImage(const TImage &      image,
                 const std::string & filename,
                 const std::string & outputComponentType,
                 bool                compress)
{
  elx::DefaultConstruct<ImageFileCastWriter<TImage>> writer;
  writer.SetInput(&image);
  writer.SetFileName(filename);
  writer.SetOutputComponentType(outputComponentType);
  writer.SetUseCompression(compress);
  writer.Update();
}

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageFileCastWriter.hxx"
#endif

#endif // itkImageFileCastWriter_h
