/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUShrinkImageFilter_h
#define __itkGPUShrinkImageFilter_h

#include "itkShrinkImageFilter.h"

#include "itkGPUImageToImageFilter.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class GPUShrinkImageFilter
* \brief Reduce the size of an image by an integer factor in each
* dimension.
*
* ShrinkImageFilter reduces the size of an image by an integer factor
* in each dimension. The algorithm implemented is a simple subsample.
* The output image size in each dimension is given by:
*
* outputSize[j] = max( vcl_floor(inputSize[j]/shrinkFactor[j]), 1 );
*
* NOTE: The physical centers of the input and output will be the
* same. Because of this, the Origin of the output may not be the same
* as the Origin of the input.
* Since this filter produces an image which is a different
* resolution, origin and with different pixel spacing than its input
* image, it needs to override several of the methods defined
* in ProcessObject in order to properly manage the pipeline execution model.
* In particular, this filter overrides
* ProcessObject::GenerateInputRequestedRegion() and
* ProcessObject::GenerateOutputInformation().
*
* This filter is implemented as a multithreaded filter.  It provides a
* ThreadedGenerateData() method for its implementation.
*
* \ingroup GeometricTransform Streamed
* \ingroup ITKImageGrid
*
* \wiki
* \wikiexample{Images/ShrinkImageFilter,Shrink an image}
* \endwiki
*/

/** Create a helper GPU Kernel class for GPUShrinkImageFilter */
itkGPUKernelClassMacro( GPUShrinkImageFilterKernel );

template< class TInputImage, class TOutputImage >
class ITK_EXPORT GPUShrinkImageFilter :
  public GPUImageToImageFilter< TInputImage, TOutputImage,
                                ShrinkImageFilter< TInputImage, TOutputImage > >
{
public:
  /** Standard class typedefs. */
  typedef GPUShrinkImageFilter                                              Self;
  typedef ShrinkImageFilter< TInputImage, TOutputImage >                    CPUSuperclass;
  typedef GPUImageToImageFilter< TInputImage, TOutputImage, CPUSuperclass > GPUSuperclass;
  typedef SmartPointer< Self >                                              Pointer;
  typedef SmartPointer< const Self >                                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUShrinkImageFilter, GPUSuperclass );

  /** Superclass typedefs. */
  typedef typename GPUSuperclass::OutputImageRegionType OutputImageRegionType;
  typedef typename GPUSuperclass::OutputImagePixelType  OutputImagePixelType;

  /** Some convenient typedefs. */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;

  typedef typename CPUSuperclass::ShrinkFactorsType ShrinkFactorsType;
  typedef typename CPUSuperclass::OutputIndexType   OutputIndexType;
  typedef typename CPUSuperclass::InputIndexType    InputIndexType;
  typedef typename CPUSuperclass::OutputOffsetType  OutputOffsetType;

  /** ImageDimension constants */
  itkStaticConstMacro( InputImageDimension, unsigned int,
                       TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
                       TOutputImage::ImageDimension );

protected:
  GPUShrinkImageFilter();
  ~GPUShrinkImageFilter(){}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  virtual void GPUGenerateData();

private:
  GPUShrinkImageFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );       // purposely not implemented

  int    m_FilterGPUKernelHandle;
  std::size_t m_DeviceLocalMemorySize;
};

/** \class GPUShrinkImageFilterFactory
* \brief Object Factory implementation for GPUShrinkImageFilter
*/
class GPUShrinkImageFilterFactory : public ObjectFactoryBase
{
public:
  typedef GPUShrinkImageFilterFactory Self;
  typedef ObjectFactoryBase           Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUShrinkImageFilter"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUShrinkImageFilterFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUShrinkImageFilterFactory::Pointer factory =
      GPUShrinkImageFilterFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUShrinkImageFilterFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );              // purposely not implemented

#define OverrideShrinkImageFilterTypeMacro( ipt, opt, dm1, dm2, dm3 )                               \
  {                                                                                                 \
    typedef Image< ipt, dm1 > InputImageType1D;                                                     \
    typedef Image< opt, dm1 > OutputImageType1D;                                                    \
    this->RegisterOverride(                                                                         \
      typeid( ShrinkImageFilter< InputImageType1D, OutputImageType1D > ).name(),                    \
      typeid( GPUShrinkImageFilter< InputImageType1D, OutputImageType1D > ).name(),                 \
      "GPU ShrinkImageFilter Override 1D",                                                          \
      true,                                                                                         \
      CreateObjectFunction< GPUShrinkImageFilter< InputImageType1D, OutputImageType1D > >::New() ); \
    typedef Image< ipt, dm2 > InputImageType2D;                                                     \
    typedef Image< opt, dm2 > OutputImageType2D;                                                    \
    this->RegisterOverride(                                                                         \
      typeid( ShrinkImageFilter< InputImageType2D, OutputImageType2D > ).name(),                    \
      typeid( GPUShrinkImageFilter< InputImageType2D, OutputImageType2D > ).name(),                 \
      "GPU ShrinkImageFilter Override 2D",                                                          \
      true,                                                                                         \
      CreateObjectFunction< GPUShrinkImageFilter< InputImageType2D, OutputImageType2D > >::New() ); \
    typedef Image< ipt, dm3 > InputImageType3D;                                                     \
    typedef Image< opt, dm3 > OutputImageType3D;                                                    \
    this->RegisterOverride(                                                                         \
      typeid( ShrinkImageFilter< InputImageType3D, OutputImageType3D > ).name(),                    \
      typeid( GPUShrinkImageFilter< InputImageType3D, OutputImageType3D > ).name(),                 \
      "GPU ShrinkImageFilter Override 3D",                                                          \
      true,                                                                                         \
      CreateObjectFunction< GPUShrinkImageFilter< InputImageType3D, OutputImageType3D > >::New() ); \
  }

  GPUShrinkImageFilterFactory()
  {
    if ( IsGPUAvailable() )
    {
      // general types
      //OverrideShrinkImageFilterTypeMacro(unsigned char, unsigned char, 1, 2,
      // 3);
      //OverrideShrinkImageFilterTypeMacro(char, char, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(unsigned short, unsigned short, 1, 2,
      // 3);
      OverrideShrinkImageFilterTypeMacro( short, short, 1, 2, 3 );
      //OverrideShrinkImageFilterTypeMacro(unsigned int, unsigned int, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(int, int, 1, 2, 3);
      OverrideShrinkImageFilterTypeMacro( float, float, 1, 2, 3 );
      //OverrideShrinkImageFilterTypeMacro(double, double, 1, 2, 3);

      // type to float, float to type
      //OverrideShrinkImageFilterTypeMacro(unsigned char, float, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(float, unsigned char, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(char, float, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(float, char, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(unsigned short, float, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(float, unsigned short, 1, 2, 3);
      OverrideShrinkImageFilterTypeMacro( short, float, 1, 2, 3 );
      //OverrideShrinkImageFilterTypeMacro(float, short, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(unsigned int, float, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(float, unsigned int, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(int, float, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(float, int, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(float, double, 1, 2, 3);
      //OverrideShrinkImageFilterTypeMacro(double, float, 1, 2, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUShrinkImageFilter.hxx"
#endif

#endif /* __itkGPUShrinkImageFilter_h */
