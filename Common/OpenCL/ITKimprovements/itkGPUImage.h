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
/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkGPUImage_h
#define itkGPUImage_h

#include "itkImage.h"
#include "itkGPUImageDataManager.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class GPUImage
 *  \brief Templated n-dimensional image class for the GPU.
 *
 * Derived from itk Image class to use with GPU image filters.
 * This class manages both CPU and GPU memory implicitly, and
 * can be used with non-GPU itk filters as well. Memory transfer
 * between CPU and GPU is done automatically and implicitly.
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/10380/3393
 *
 * \ingroup ITKGPUCommon
 */
template <typename TPixel, unsigned int VImageDimension = 2>
class ITK_TEMPLATE_EXPORT ITKOpenCL_EXPORT GPUImage : public Image<TPixel, VImageDimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUImage);

  using Self = GPUImage;
  using Superclass = Image<TPixel, VImageDimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;
  using ConstWeakPointer = WeakPointer<const Self>;

  itkNewMacro(Self);

  itkTypeMacro(GPUImage, Image);

  itkStaticConstMacro(ImageDimension, unsigned int, VImageDimension);

  using typename Superclass::PixelType;
  using typename Superclass::ValueType;
  using typename Superclass::InternalPixelType;
  using typename Superclass::IOPixelType;
  using typename Superclass::DirectionType;
  using typename Superclass::SpacingType;
  using typename Superclass::PixelContainer;
  using typename Superclass::SizeType;
  using typename Superclass::IndexType;
  using typename Superclass::OffsetType;
  using typename Superclass::RegionType;
  using PixelContainerPointer = typename PixelContainer::Pointer;
  using PixelContainerConstPointer = typename PixelContainer::ConstPointer;
  using typename Superclass::AccessorType;

  using AccessorFunctorType = DefaultPixelAccessorFunctor<Self>;

  using NeighborhoodAccessorFunctorType = NeighborhoodAccessorFunctor<Self>;

  /** Override Rebind and RebindImageType of itk::Image class */
  template <typename UPixelType, unsigned int VUImageDimension = VImageDimension>
  struct Rebind
  {
    using Type = itk::GPUImage<UPixelType, VUImageDimension>;
  };

  template <typename UPixelType, unsigned int VUImageDimension = VImageDimension>
  using RebindImageType = itk::GPUImage<UPixelType, VUImageDimension>;


  /** Allocate CPU and GPU memory space */
  void
  Allocate(bool initialize = false) override;

  void
  AllocateGPU();

  void
  Initialize() override;

  void
  FillBuffer(const TPixel & value);

  void
  SetPixel(const IndexType & index, const TPixel & value);

  const TPixel &
  GetPixel(const IndexType & index) const;

  TPixel &
  GetPixel(const IndexType & index);

  const TPixel & operator[](const IndexType & index) const;

  TPixel & operator[](const IndexType & index);

  /** Explicit synchronize CPU/GPU buffers */
  void
  UpdateBuffers();

  /** Explicit synchronize CPU/GPU buffers */
  void
  UpdateCPUBuffer();

  void
  UpdateGPUBuffer();

  /** Get CPU buffer pointer */
  TPixel *
  GetBufferPointer() override;

  const TPixel *
  GetBufferPointer() const override;

  /** Return the Pixel Accessor object */
  AccessorType
  GetPixelAccessor()
  {
    m_DataManager->SetGPUBufferDirty();
    return Superclass::GetPixelAccessor();
  }


  /** Return the Pixel Accesor object */
  const AccessorType
  GetPixelAccessor() const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelAccessor();
  }


  /** Return the NeighborhoodAccessor functor */
  NeighborhoodAccessorFunctorType
  GetNeighborhoodAccessor()
  {
    m_DataManager->SetGPUBufferDirty();
    return NeighborhoodAccessorFunctorType();
  }


  /** Return the NeighborhoodAccessor functor */
  const NeighborhoodAccessorFunctorType
  GetNeighborhoodAccessor() const
  {
    m_DataManager->UpdateCPUBuffer();
    return NeighborhoodAccessorFunctorType();
  }


  void
  SetPixelContainer(PixelContainer * container);

  /** Return a pointer to the container. */
  PixelContainer *
  GetPixelContainer()
  {
    m_DataManager->SetGPUBufferDirty();
    return Superclass::GetPixelContainer();
  }


  const PixelContainer *
  GetPixelContainer() const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelContainer();
  }


  void
  SetCurrentCommandQueue(int queueid)
  {
    m_DataManager->SetCurrentCommandQueue(queueid);
  }


  int
  GetCurrentCommandQueueId()
  {
    return m_DataManager->GetCurrentCommandQueueId();
  }


  GPUDataManager::Pointer
  GetGPUDataManager() const;

  /** Override DataHasBeenGenerated() in DataObject class.
   * We need this because CPU time stamp is always bigger
   * than GPU's. That is because Modified() is called at
   * the end of each filter in the pipeline so although we
   * increment GPU's time stamp in GPUGenerateData() the
   * CPU's time stamp will be increased after that.
   */
  void
  DataHasBeenGenerated() override
  {
    Superclass::DataHasBeenGenerated();

    if (m_DataManager->IsCPUBufferDirty())
    {
      m_DataManager->Modified();
    }
  }


  /** Graft the data and information from one GPUImage to another. */
  void
  Graft(const DataObject * data) override;

  void
  GraftITKImage(const DataObject * data);

  /** Whenever the image has been modified, set the GPU Buffer to dirty */
  void
  Modified() const override;

  /** Get matrices intended to help with the conversion of Index coordinates
   *  to PhysicalPoint coordinates */
  itkGetConstReferenceMacro(IndexToPhysicalPoint, DirectionType);
  itkGetConstReferenceMacro(PhysicalPointToIndex, DirectionType);

protected:
  GPUImage();
  ~GPUImage() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  bool m_Graft;

  typename GPUImageDataManager<GPUImage>::Pointer m_DataManager;
};

//------------------------------------------------------------------------------
template <typename T>
class ITK_TEMPLATE_EXPORT GPUTraits
{
public:
  using Type = T;
};

template <typename TPixelType, unsigned int NDimension>
class ITK_TEMPLATE_EXPORT GPUTraits<Image<TPixelType, NDimension>>
{
public:
  using Type = GPUImage<TPixelType, NDimension>;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUImage.hxx"
#endif

#endif
