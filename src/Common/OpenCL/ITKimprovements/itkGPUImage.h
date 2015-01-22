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
#ifndef __itkGPUImage_h
#define __itkGPUImage_h

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
template< typename TPixel, unsigned int VImageDimension = 2 >
class ITKOpenCL_EXPORT GPUImage : public Image< TPixel, VImageDimension >
{
public:

  typedef GPUImage                         Self;
  typedef Image< TPixel, VImageDimension > Superclass;
  typedef SmartPointer< Self >             Pointer;
  typedef SmartPointer< const Self >       ConstPointer;
  typedef WeakPointer< const Self >        ConstWeakPointer;

  itkNewMacro( Self );

  itkTypeMacro( GPUImage, Image );

  itkStaticConstMacro( ImageDimension, unsigned int, VImageDimension );

  typedef typename Superclass::PixelType         PixelType;
  typedef typename Superclass::ValueType         ValueType;
  typedef typename Superclass::InternalPixelType InternalPixelType;
  typedef typename Superclass::IOPixelType       IOPixelType;
  typedef typename Superclass::DirectionType     DirectionType;
  typedef typename Superclass::SpacingType       SpacingType;
  typedef typename Superclass::PixelContainer    PixelContainer;
  typedef typename Superclass::SizeType          SizeType;
  typedef typename Superclass::IndexType         IndexType;
  typedef typename Superclass::OffsetType        OffsetType;
  typedef typename Superclass::RegionType        RegionType;
  typedef typename PixelContainer::Pointer       PixelContainerPointer;
  typedef typename PixelContainer::ConstPointer  PixelContainerConstPointer;
  typedef typename Superclass::AccessorType      AccessorType;

  typedef DefaultPixelAccessorFunctor< Self > AccessorFunctorType;

  typedef NeighborhoodAccessorFunctor< Self > NeighborhoodAccessorFunctorType;

  /** Allocate CPU and GPU memory space */
  virtual void Allocate( bool initialize = false ) ITK_OVERRIDE;

  void AllocateGPU( void );

  virtual void Initialize( void );

  void FillBuffer( const TPixel & value );

  void SetPixel( const IndexType & index, const TPixel & value );

  const TPixel & GetPixel( const IndexType & index ) const;

  TPixel & GetPixel( const IndexType & index );

  const TPixel & operator[]( const IndexType & index ) const;

  TPixel & operator[]( const IndexType & index );

  /** Explicit synchronize CPU/GPU buffers */
  void UpdateBuffers( void );

  /** Explicit synchronize CPU/GPU buffers */
  void UpdateCPUBuffer( void );

  void UpdateGPUBuffer( void );

  /** Get CPU buffer pointer */
  TPixel * GetBufferPointer( void );

  const TPixel * GetBufferPointer( void ) const;

  /** Return the Pixel Accessor object */
  AccessorType GetPixelAccessor( void )
  {
    m_DataManager->SetGPUBufferDirty();
    return Superclass::GetPixelAccessor();
  }


  /** Return the Pixel Accesor object */
  const AccessorType GetPixelAccessor( void ) const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelAccessor();
  }


  /** Return the NeighborhoodAccessor functor */
  NeighborhoodAccessorFunctorType GetNeighborhoodAccessor( void )
  {
    m_DataManager->SetGPUBufferDirty();
    //return Superclass::GetNeighborhoodAccessor();
    return NeighborhoodAccessorFunctorType();
  }


  /** Return the NeighborhoodAccessor functor */
  const NeighborhoodAccessorFunctorType GetNeighborhoodAccessor( void ) const
  {
    m_DataManager->UpdateCPUBuffer();
    //return Superclass::GetNeighborhoodAccessor();
    return NeighborhoodAccessorFunctorType();
  }


  void SetPixelContainer( PixelContainer * container );

  /** Return a pointer to the container. */
  PixelContainer * GetPixelContainer( void )
  {
    m_DataManager->SetGPUBufferDirty(); return Superclass::GetPixelContainer();
  }


  const PixelContainer * GetPixelContainer( void ) const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelContainer();
  }


  void SetCurrentCommandQueue( int queueid )
  {
    m_DataManager->SetCurrentCommandQueue( queueid );
  }


  int GetCurrentCommandQueueId( void )
  {
    return m_DataManager->GetCurrentCommandQueueId();
  }


  GPUDataManager::Pointer GetGPUDataManager( void ) const;

  /** Override DataHasBeenGenerated() in DataObject class.
   * We need this because CPU time stamp is always bigger
   * than GPU's. That is because Modified() is called at
   * the end of each filter in the pipeline so although we
   * increment GPU's time stamp in GPUGenerateData() the
   * CPU's time stamp will be increased after that.
   */
  void DataHasBeenGenerated( void )
  {
    Superclass::DataHasBeenGenerated();

    if( m_DataManager->IsCPUBufferDirty() )
    {
      m_DataManager->Modified();
    }

  }


  /** Graft the data and information from one GPUImage to another. */
  virtual void Graft( const DataObject * data );

  void GraftITKImage( const DataObject * data );

  /** Whenever the image has been modified, set the GPU Buffer to dirty */
  virtual void Modified( void ) const;

  /** Get matrices intended to help with the conversion of Index coordinates
   *  to PhysicalPoint coordinates */
  itkGetConstReferenceMacro( IndexToPhysicalPoint, DirectionType );
  itkGetConstReferenceMacro( PhysicalPointToIndex, DirectionType );

protected:

  GPUImage();
  virtual ~GPUImage() {}

  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  GPUImage( const Self & );       // purposely not implemented
  void operator=( const Self & ); // purposely not implemented

  bool m_Graft;

  typename GPUImageDataManager< GPUImage >::Pointer m_DataManager;
};

//------------------------------------------------------------------------------
template< typename T >
class GPUTraits
{
public:

  typedef T Type;
};

template< typename TPixelType, unsigned int NDimension >
class GPUTraits< Image< TPixelType, NDimension > >
{
public:

  typedef GPUImage< TPixelType, NDimension > Type;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUImage.hxx"
#endif

#endif
