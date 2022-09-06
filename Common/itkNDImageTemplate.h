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
#ifndef itkNDImageTemplate_h
#define itkNDImageTemplate_h

#include "itkNDImageBase.h"
#include "itkImageFileReader.h"

namespace itk
{

/**
 * \class NDImageTemplate
 * \brief This class is a specialization of the NDImageBase,
 * which acts as a wrap around an itk::Image.
 *
 * The NDImageTemplate class is a kind of wrap around the
 * itk::Image. It has an itk::Image object as an internal
 * member variable. Most functions simply call the
 * the corresponding function in the itk::Object. For some
 * functions, the in/output arguments have to be converted
 * from/to arrays with runtime length to/from arrays with
 * compile time length.
 *
 * \sa NDImageBase
 * \ingroup Miscellaneous
 */

template <class TPixel, unsigned int VDimension>
class ITK_TEMPLATE_EXPORT NDImageTemplate : public NDImageBase<TPixel>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(NDImageTemplate);

  /** Standard class typedefs.*/
  using Self = NDImageTemplate;
  using Superclass = NDImageBase<TPixel>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NDImageTemplate, NDImageBase);

  /**
   * Typedefs inherited from Superclass.
   */

  itkStaticConstMacro(Dimension, unsigned int, VDimension);

  using typename Superclass::DataObjectPointer;

  /** Type definitions like normal itkImages, independent of the dimension */
  using typename Superclass::PixelType;
  using typename Superclass::ValueType;
  using typename Superclass::InternalPixelType;
  using typename Superclass::AccessorType;
  using typename Superclass::PixelContainer;
  using typename Superclass::PixelContainerPointer;
  using typename Superclass::PixelContainerConstPointer;

  using typename Superclass::SpacingValueType;
  using typename Superclass::PointValueType;
  using typename Superclass::IndexValueType;
  using typename Superclass::SizeValueType;
  using typename Superclass::OffsetValueType;

  /** ND versions of the index and sizetypes etc. */
  using typename Superclass::IndexType;
  using typename Superclass::SizeType;
  using typename Superclass::SpacingType;
  using typename Superclass::PointType;
  using typename Superclass::OffsetType;

  /** Typedefs dependent on the dimension */
  using ImageType = Image<TPixel, VDimension>;
  using ImagePointer = typename ImageType::Pointer;
  using WriterType = ImageFileWriter<ImageType>;
  using WriterPointer = typename WriterType::Pointer;
  using ReaderType = ImageFileReader<ImageType>;
  using ReaderPointer = typename ReaderType::Pointer;

  /** Original, itk, versions of the index and sizetypes etc. */
  using IndexTypeD = typename ImageType::IndexType;
  using SizeTypeD = typename ImageType::SizeType;
  using SpacingTypeD = typename ImageType::SpacingType;
  using PointTypeD = typename ImageType::PointType;
  using OffsetTypeD = typename ImageType::OffsetType;

  void
  SetRegions(SizeType size) override;

  void
  SetRequestedRegion(DataObject * data) override;

  void
  Allocate() override;

  void
  Initialize() override;

  void
  FillBuffer(const TPixel & value) override;

  void
  SetPixel(const IndexType & index, const TPixel & value) override;

  const TPixel &
  GetPixel(const IndexType & index) const override;

  TPixel &
  GetPixel(const IndexType & index) override;

  TPixel *
  GetBufferPointer() override;

  const TPixel *
  GetBufferPointer() const override;

  PixelContainer *
  GetPixelContainer() override;

  const PixelContainer *
  GetPixelContainer() const override;

  void
  SetPixelContainer(PixelContainer * container) override;

  AccessorType
  GetPixelAccessor() override;

  const AccessorType
  GetPixelAccessor() const override;

  void
  SetSpacing(const SpacingType & spacing) override;

  void
  SetOrigin(const PointType & origin) override;

  SpacingType
  GetSpacing() override;

  PointType
  GetOrigin() override;

  /** \todo Transform IndexToPoint methods. */

  void
  CopyInformation(const DataObject * data) override;

  const OffsetValueType *
  GetOffsetTable() const override;

  OffsetValueType
  ComputeOffset(const IndexType & ind) const override;

  IndexType
  ComputeIndex(OffsetValueType offset) const override;

  /** Extra functions for NDImage. */

  /** Get the Dimension.*/
  unsigned int
  ImageDimension() override;

  unsigned int
  GetImageDimension() override;

  /** Get the actual image */
  itkGetModifiableObjectMacro(Image, DataObject);
  itkGetModifiableObjectMacro(Writer, ProcessObject);
  itkGetModifiableObjectMacro(Reader, ProcessObject);

  /** Write the actual image to file. */
  void
  Write() override;

  /** Read image data from file into the actual image */
  void
  Read() override;

  /** Use New method to create a new actual image */
  void
  CreateNewImage() override;

  void
  SetImageIOWriter(ImageIOBase * _arg) override;

  ImageIOBase *
  GetImageIOWriter() override;

  void
  SetImageIOReader(ImageIOBase * _arg) override;

  ImageIOBase *
  GetImageIOReader() override;

  /** Set/Get the Output/Input FileName */
  void
  SetOutputFileName(const char * name) override;

  void
  SetInputFileName(const char * name) override;

  const char *
  GetOutputFileName() override;

  const char *
  GetInputFileName() override;

protected:
  NDImageTemplate() = default;
  ~NDImageTemplate() override = default;

  // virtual void PrintSelf(std::ostream& os, Indent indent) const;

  ImagePointer  m_Image{ nullptr };
  WriterPointer m_Writer{ nullptr };
  ReaderPointer m_Reader{ nullptr };

  template <class TIn, class TOut>
  class ITK_TEMPLATE_EXPORT ConvertToDynamicArray
  {
  public:
    inline static TOut
    DO(const TIn & in)
    {
      TOut out(VDimension);

      for (unsigned int i = 0; i < VDimension; ++i)
      {
        out[i] = in[i];
      }
      return out;
    }
  };

  template <class TIn, class TOut>
  class ITK_TEMPLATE_EXPORT ConvertToStaticArray
  {
  public:
    inline static TOut
    DO(const TIn & in)
    {
      TOut out;

      for (unsigned int i = 0; i < VDimension; ++i)
      {
        out[i] = in[i];
      }
      return out;
    }
  };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkNDImageTemplate.hxx"
#endif

#endif // end #ifndef itkNDImageTemplate_h
