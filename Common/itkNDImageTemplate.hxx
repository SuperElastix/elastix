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

#ifndef itkNDImageTemplate_hxx
#define itkNDImageTemplate_hxx

#include "itkNDImageTemplate.h"

namespace itk
{

template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetRegions(SizeType size)
{
  this->m_Image->SetRegions(ConvertToStaticArray<SizeType, SizeTypeD>(size));
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetRequestedRegion(DataObject * data)
{
  this->m_Image->SetRequestedRegion(data);
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::Allocate()
{
  this->m_Image->Allocate();
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::Initialize()
{
  this->m_Image->Initialize();
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::FillBuffer(const TPixel & value)
{
  this->m_Image->FillBuffer(value);
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetPixel(const IndexType & index, const TPixel & value)
{
  this->m_Image->SetPixel(ConvertToStaticArray<IndexType, IndexTypeD>(index), value);
}


template <typename TPixel, unsigned int VDimension>
const TPixel &
NDImageTemplate<TPixel, VDimension>::GetPixel(const IndexType & index) const
{
  return this->m_Image->GetPixel(ConvertToStaticArray<IndexType, IndexTypeD>(index));
}


template <typename TPixel, unsigned int VDimension>
TPixel &
NDImageTemplate<TPixel, VDimension>::GetPixel(const IndexType & index)
{
  return this->m_Image->GetPixel(ConvertToStaticArray<IndexType, IndexTypeD>(index));
}


template <typename TPixel, unsigned int VDimension>
TPixel *
NDImageTemplate<TPixel, VDimension>::GetBufferPointer()
{
  return this->m_Image->GetBufferPointer();
}


template <typename TPixel, unsigned int VDimension>
const TPixel *
NDImageTemplate<TPixel, VDimension>::GetBufferPointer() const
{
  return this->m_Image->GetBufferPointer();
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetPixelContainer() -> PixelContainer *
{
  return this->m_Image->GetPixelContainer();
}

template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetPixelContainer() const -> const PixelContainer *
{
  return this->m_Image->GetPixelContainer();
}

template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetPixelContainer(PixelContainer * container)
{
  this->m_Image->SetPixelContainer(container);
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetPixelAccessor() -> AccessorType
{
  return this->m_Image->GetPixelAccessor();
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetPixelAccessor() const -> const AccessorType
{
  return this->m_Image->GetPixelAccessor();
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetSpacing(const SpacingType & spacing)
{
  this->m_Image->SetSpacing(ConvertToStaticArray<SpacingType, SpacingTypeD>(spacing));
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetOrigin(const PointType & origin)
{
  this->m_Image->SetOrigin(ConvertToStaticArray<PointType, PointTypeD>(origin));
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetSpacing() -> SpacingType
{
  return ConvertToDynamicArray<SpacingTypeD, SpacingType>(this->m_Image->GetSpacing());
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetOrigin() -> PointType
{
  return ConvertToDynamicArray<PointTypeD, PointType>(this->m_Image->GetOrigin());
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::CopyInformation(const DataObject * data)
{
  this->m_Image->CopyInformation(data);
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::GetOffsetTable() const -> const OffsetValueType *
{
  return this->m_Image->GetOffsetTable();
}

template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::ComputeOffset(const IndexType & ind) const -> OffsetValueType
{
  return this->m_Image->ComputeOffset(ConvertToStaticArray<IndexType, IndexTypeD>(ind));
}


template <typename TPixel, unsigned int VDimension>
auto
NDImageTemplate<TPixel, VDimension>::ComputeIndex(OffsetValueType offset) const -> IndexType
{
  return ConvertToDynamicArray<IndexTypeD, IndexType>(this->m_Image->ComputeIndex(offset));
}


template <typename TPixel, unsigned int VDimension>
unsigned int
NDImageTemplate<TPixel, VDimension>::ImageDimension()
{
  return VDimension;
}


template <typename TPixel, unsigned int VDimension>
unsigned int
NDImageTemplate<TPixel, VDimension>::GetImageDimension()
{
  return VDimension;
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::Write()
{
  if (this->m_Writer)
  {
    this->m_Writer->SetInput(this->m_Image);
    this->m_Writer->Write();
  }
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::Read()
{
  if (this->m_Reader)
  {
    this->m_Reader->Update();
    this->m_Image = this->m_Reader->GetOutput();
  }
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::CreateNewImage()
{
  this->m_Image = ImageType::New();
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetImageIOWriter(ImageIOBase * _arg)
{
  if (!(this->m_Writer))
  {
    this->m_Writer = WriterType::New();
  }
  this->m_Writer->SetImageIO(_arg);
}


template <typename TPixel, unsigned int VDimension>
ImageIOBase *
NDImageTemplate<TPixel, VDimension>::GetImageIOWriter()
{
  if (this->m_Writer)
  {
    return this->m_Writer->GetModifiableImageIO();
  }
  else
  {
    return nullptr;
  }
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetImageIOReader(ImageIOBase * _arg)
{
  if (!(this->m_Reader))
  {
    this->m_Reader = ReaderType::New();
  }
  this->m_Reader->SetImageIO(_arg);
}


template <typename TPixel, unsigned int VDimension>
ImageIOBase *
NDImageTemplate<TPixel, VDimension>::GetImageIOReader()
{
  if (this->m_Reader)
  {
    return this->m_Reader->GetModifiableImageIO();
  }
  else
  {
    return nullptr;
  }
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetOutputFileName(const char * name)
{
  if (!(this->m_Writer))
  {
    this->m_Writer = WriterType::New();
  }
  this->m_Writer->SetFileName(name);
}


template <typename TPixel, unsigned int VDimension>
void
NDImageTemplate<TPixel, VDimension>::SetInputFileName(const char * name)
{
  if (!(this->m_Reader))
  {
    this->m_Reader = ReaderType::New();
  }
  this->m_Reader->SetFileName(name);
}


template <typename TPixel, unsigned int VDimension>
const char *
NDImageTemplate<TPixel, VDimension>::GetOutputFileName()
{
  if (this->m_Writer)
  {
    return this->m_Writer->GetFileName();
  }
  else
  {
    return "";
  }
}


template <typename TPixel, unsigned int VDimension>
const char *
NDImageTemplate<TPixel, VDimension>::GetInputFileName()
{
  if (this->m_Reader)
  {
    return this->m_Reader->GetFileName().c_str();
  }
  else
  {
    return "";
  }
}


} // end namespace itk

#endif // end #ifndef itkNDImageTemplate_hxx
