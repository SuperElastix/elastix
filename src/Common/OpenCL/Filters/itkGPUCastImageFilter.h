/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUCastImageFilter_h
#define __itkGPUCastImageFilter_h

#include "itkCastImageFilter.h"
#include "itkSimpleDataObjectDecorator.h"

#include "itkOpenCLUtil.h"
#include "itkGPUFunctorBase.h"
#include "itkGPUKernelManager.h"
#include "itkGPUUnaryFunctorImageFilter.h"

namespace itk
{
/** \class GPUCastImageFilter
 *
 * \brief GPU version of cast image filter.
 *
 * \ingroup GPUCommon
 */
namespace Functor
{
template< class TInput, class TOutput >
class GPUCast : public GPUFunctorBase
{
public:
  GPUCast() {}
  ~GPUCast() {}

  /** Setup GPU kernel arguments for this functor.
   * Returns current argument index to set additional arguments in the GPU kernel.
   */
  int SetGPUKernelArguments( GPUKernelManager::Pointer KernelManager, int KernelHandle )
  {
    return 0;
  }
};
} // end of namespace Functor

/** Create a helper GPU Kernel class for GPUCastImageFilter */
itkGPUKernelClassMacro( GPUCastImageFilterKernel );

/** GPUCastImageFilter class definition */
template< class TInputImage, class TOutputImage >
class ITK_EXPORT GPUCastImageFilter :
  public GPUUnaryFunctorImageFilter<
    TInputImage, TOutputImage,
    Functor::GPUCast< typename TInputImage::PixelType,
                      typename TOutputImage::PixelType >,
    CastImageFilter< TInputImage, TOutputImage > >
{
public:
  /** Standard class typedefs. */
  typedef GPUCastImageFilter Self;
  typedef GPUUnaryFunctorImageFilter<
      TInputImage, TOutputImage,
      Functor::GPUCast<
        typename TInputImage::PixelType,
        typename TOutputImage::PixelType >,
      CastImageFilter< TInputImage, TOutputImage > >   GPUSuperclass;
  typedef CastImageFilter< TInputImage, TOutputImage > CPUSuperclass;
  typedef SmartPointer< Self >                         Pointer;
  typedef SmartPointer< const Self >                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCastImageFilter, GPUUnaryFunctorImageFilter );

  /** Pixel types. */
  typedef typename TInputImage::PixelType  InputPixelType;
  typedef typename TOutputImage::PixelType OutputPixelType;

  /** Type of DataObjects to use for scalar inputs */
  typedef SimpleDataObjectDecorator< InputPixelType > InputPixelObjectType;

protected:
  GPUCastImageFilter();
  virtual ~GPUCastImageFilter() {}

  /** Unlike CPU version, GPU version of binary threshold filter is not
  multi-threaded */
  virtual void GPUGenerateData( void );

private:
  GPUCastImageFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );     // purposely not implemented
};

/** Object Factory implementation for GPUCastImageFilter */
class GPUCastImageFilterFactory : public ObjectFactoryBase
{
public:
  typedef GPUCastImageFilterFactory  Self;
  typedef ObjectFactoryBase          Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const
  {
    return ITK_SOURCE_VERSION;
  }

  const char * GetDescription() const
  {
    return "A Factory for GPUCastImageFilter";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCastImageFilterFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUCastImageFilterFactory::Pointer factory =
      GPUCastImageFilterFactory::New();
    itk::ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUCastImageFilterFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );            // purposely not implemented

// Override default
#define OverrideCastFilterTypeMacro( ipt, opt, dm1, dm2, dm3 )                                         \
  {                                                                                                    \
    typedef itk::Image< ipt, dm1 > InputImageType1D;                                                   \
    typedef itk::Image< opt, dm1 > OutputImageType1D;                                                  \
    this->RegisterOverride(                                                                            \
      typeid( itk::CastImageFilter< InputImageType1D, OutputImageType1D > ).name(),                    \
      typeid( itk::GPUCastImageFilter< InputImageType1D, OutputImageType1D > ).name(),                 \
      "GPU Cast Image Filter Override 1D",                                                             \
      true,                                                                                            \
      itk::CreateObjectFunction< GPUCastImageFilter< InputImageType1D, OutputImageType1D > >::New() ); \
    typedef itk::Image< ipt, dm2 > InputImageType2D;                                                   \
    typedef itk::Image< opt, dm2 > OutputImageType2D;                                                  \
    this->RegisterOverride(                                                                            \
      typeid( itk::CastImageFilter< InputImageType2D, OutputImageType2D > ).name(),                    \
      typeid( itk::GPUCastImageFilter< InputImageType2D, OutputImageType2D > ).name(),                 \
      "GPU Cast Image Filter Override 2D",                                                             \
      true,                                                                                            \
      itk::CreateObjectFunction< GPUCastImageFilter< InputImageType2D, OutputImageType2D > >::New() ); \
    typedef itk::Image< ipt, dm3 > InputImageType3D;                                                   \
    typedef itk::Image< opt, dm3 > OutputImageType3D;                                                  \
    this->RegisterOverride(                                                                            \
      typeid( itk::CastImageFilter< InputImageType3D, OutputImageType3D > ).name(),                    \
      typeid( itk::GPUCastImageFilter< InputImageType3D, OutputImageType3D > ).name(),                 \
      "GPU Cast Image Filter Override 3D",                                                             \
      true,                                                                                            \
      itk::CreateObjectFunction< GPUCastImageFilter< InputImageType3D, OutputImageType3D > >::New() ); \
  }

// Override when itkGPUImage is second template argument
#define OverrideCastFilterGPUOutTypeMacro( ipt, opt, dm1, dm2, dm3 )                                      \
  {                                                                                                       \
    typedef itk::Image< ipt, dm1 >    InputImageType1D;                                                   \
    typedef itk::GPUImage< opt, dm1 > GPUOutputImageType1D;                                               \
    this->RegisterOverride(                                                                               \
      typeid( itk::CastImageFilter< InputImageType1D, GPUOutputImageType1D > ).name(),                    \
      typeid( itk::GPUCastImageFilter< InputImageType1D, GPUOutputImageType1D > ).name(),                 \
      "GPU Cast Image Filter Override 1D",                                                                \
      true,                                                                                               \
      itk::CreateObjectFunction< GPUCastImageFilter< InputImageType1D, GPUOutputImageType1D > >::New() ); \
    typedef itk::Image< ipt, dm2 >    InputImageType2D;                                                   \
    typedef itk::GPUImage< opt, dm2 > GPUOutputImageType2D;                                               \
    this->RegisterOverride(                                                                               \
      typeid( itk::CastImageFilter< InputImageType2D, GPUOutputImageType2D > ).name(),                    \
      typeid( itk::GPUCastImageFilter< InputImageType2D, GPUOutputImageType2D > ).name(),                 \
      "GPU Cast Image Filter Override 2D",                                                                \
      true,                                                                                               \
      itk::CreateObjectFunction< GPUCastImageFilter< InputImageType2D, GPUOutputImageType2D > >::New() ); \
    typedef itk::Image< ipt, dm3 >    InputImageType3D;                                                   \
    typedef itk::GPUImage< opt, dm3 > GPUOutputImageType3D;                                               \
    this->RegisterOverride(                                                                               \
      typeid( itk::CastImageFilter< InputImageType3D, GPUOutputImageType3D > ).name(),                    \
      typeid( itk::GPUCastImageFilter< InputImageType3D, GPUOutputImageType3D > ).name(),                 \
      "GPU Cast Image Filter Override 3D",                                                                \
      true,                                                                                               \
      itk::CreateObjectFunction< GPUCastImageFilter< InputImageType3D, GPUOutputImageType3D > >::New() ); \
  }

  GPUCastImageFilterFactory()
  {
    if ( IsGPUAvailable() )
    {
      // general types
      //OverrideCastFilterTypeMacro(unsigned char, unsigned char, 1, 2, 3);
      //OverrideCastFilterTypeMacro(char, char, 1, 2, 3);
      //OverrideCastFilterTypeMacro(unsigned short, unsigned short, 1, 2, 3);
      OverrideCastFilterTypeMacro( short, short, 1, 2, 3 );
      //OverrideCastFilterTypeMacro(unsigned int, unsigned int, 1, 2, 3);
      //OverrideCastFilterTypeMacro(int, int, 1, 2, 3);
      OverrideCastFilterTypeMacro( float, float, 1, 2, 3 );
      //OverrideCastFilterTypeMacro(double, double, 1, 2, 3);

      // type to float, float to type
      //OverrideCastFilterTypeMacro(unsigned char, float, 1, 2, 3);
      //OverrideCastFilterTypeMacro(float, unsigned char, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(unsigned char, float, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(float, unsigned char, 1, 2, 3);

      //OverrideCastFilterTypeMacro(char, float, 1, 2, 3);
      //OverrideCastFilterTypeMacro(float, char, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(char, float, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(float, char, 1, 2, 3);

      //OverrideCastFilterTypeMacro(unsigned short, float, 1, 2, 3);
      //OverrideCastFilterTypeMacro(float, unsigned short, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(unsigned short, float, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(float, unsigned short, 1, 2, 3);

      OverrideCastFilterTypeMacro( short, float, 1, 2, 3 );
      OverrideCastFilterTypeMacro( float, short, 1, 2, 3 );
      OverrideCastFilterGPUOutTypeMacro( short, float, 1, 2, 3 );
      OverrideCastFilterGPUOutTypeMacro( float, short, 1, 2, 3 );

      //OverrideCastFilterTypeMacro(unsigned int, float, 1, 2, 3);
      //OverrideCastFilterTypeMacro(float, unsigned int, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(unsigned int, float, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(float, unsigned int, 1, 2, 3);

      //OverrideCastFilterTypeMacro(int, float, 1, 2, 3);
      //OverrideCastFilterTypeMacro(float, int, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(int, float, 1, 2, 3);
      //OverrideCastFilterGPUOutTypeMacro(float, int, 1, 2, 3);

      OverrideCastFilterTypeMacro( double, float, 1, 2, 3 );
      OverrideCastFilterTypeMacro( float, double, 1, 2, 3 );
      OverrideCastFilterGPUOutTypeMacro( double, float, 1, 2, 3 );
      OverrideCastFilterGPUOutTypeMacro( float, double, 1, 2, 3 );
    }
  }
};
} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUCastImageFilter.hxx"
#endif

#endif /* __itkGPUCastImageFilter_h */
