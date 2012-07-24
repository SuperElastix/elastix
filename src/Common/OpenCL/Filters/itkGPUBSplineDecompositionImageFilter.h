/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineDecompositionImageFilter_h
#define __itkGPUBSplineDecompositionImageFilter_h

#include "itkBSplineDecompositionImageFilter.h"

#include "itkGPUInPlaceImageFilter.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class GPUBSplineDecompositionImageFilter
* \brief Calculates the B-Spline coefficients of an image. Spline order may be from 0 to 5.
*
* This class defines N-Dimension B-Spline transformation.
* It is based on:
*    [1] M. Unser,
*       "Splines: A Perfect Fit for Signal and Image Processing,"
*        IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22-38,
*        November 1999.
*    [2] M. Unser, A. Aldroubi and M. Eden,
*        "B-Spline Signal Processing: Part I--Theory,"
*        IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 821-832,
*        February 1993.
*    [3] M. Unser, A. Aldroubi and M. Eden,
*        "B-Spline Signal Processing: Part II--Efficient Design and Applications,"
*        IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 834-848,
*        February 1993.
* And code obtained from bigwww.epfl.ch by Philippe Thevenaz
*
* Limitations:  Spline order must be between 0 and 5.
*               Spline order must be set before setting the image.
*               Uses mirror boundary conditions.
*               Requires the same order of Spline for each dimension.
*               Can only process LargestPossibleRegion
*
* \sa itkBSplineInterpolateImageFunction
* \see BSplineDecompositionImageFilter
* \ingroup ITK-GPUCommon
*/
template< class TInputImage, class TOutputImage >
class ITK_EXPORT GPUBSplineDecompositionImageFilter
  : public GPUImageToImageFilter< TInputImage, TOutputImage,
  BSplineDecompositionImageFilter< TInputImage, TOutputImage > >
{
public:
  /** Standard class typedefs. */
  typedef GPUBSplineDecompositionImageFilter  Self;
  typedef BSplineDecompositionImageFilter< TInputImage, TOutputImage > CPUSuperclass;
  typedef GPUImageToImageFilter< TInputImage, TOutputImage, CPUSuperclass > GPUSuperclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineDecompositionImageFilter, GPUSuperclass);

  /** Superclass typedefs. */
  typedef typename GPUSuperclass::OutputImageRegionType OutputImageRegionType;
  typedef typename GPUSuperclass::OutputImagePixelType  OutputImagePixelType;
  //typedef typename CPUSuperclass::ScalarRealType        ScalarRealType;

  /** Some convenient typedefs. */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
    TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
    TOutputImage::ImageDimension);

protected:
  GPUBSplineDecompositionImageFilter();
  ~GPUBSplineDecompositionImageFilter(){};

  virtual void GPUGenerateData();
  virtual void PrintSelf(std::ostream & os, Indent indent) const;

private:
  GPUBSplineDecompositionImageFilter(const Self &);  //purposely not implemented
  void operator=(const Self &);                   //purposely not implemented

  int m_FilterGPUKernelHandle;
  size_t m_DeviceLocalMemorySize;
};

/** \class GPUBSplineDecompositionImageFilterFactory
* \brief Object Factory implementation for GPUBSplineDecompositionImageFilter
*/
class GPUBSplineDecompositionImageFilterFactory : public ObjectFactoryBase
{
public:
  typedef GPUBSplineDecompositionImageFilterFactory Self;
  typedef ObjectFactoryBase                         Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char* GetDescription() const { return "A Factory for GPUBSplineDecompositionImageFilter"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineDecompositionImageFilterFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    GPUBSplineDecompositionImageFilterFactory::Pointer factory = GPUBSplineDecompositionImageFilterFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  GPUBSplineDecompositionImageFilterFactory(const Self&); //purposely not implemented
  void operator=(const Self&);                            //purposely not implemented

#define OverrideBSplineDecompositionImageFilterTypeMacro(ipt,opt,dm1,dm2,dm3)\
  {\
  typedef Image<ipt,dm1> InputImageType1D;\
  typedef Image<opt,dm1> OutputImageType1D;\
  this->RegisterOverride(\
  typeid(BSplineDecompositionImageFilter<InputImageType1D,OutputImageType1D>).name(),\
  typeid(GPUBSplineDecompositionImageFilter<InputImageType1D,OutputImageType1D>).name(),\
  "GPU BSplineDecompositionImageFilter Override 1D",\
  true,\
  CreateObjectFunction<GPUBSplineDecompositionImageFilter<InputImageType1D,OutputImageType1D> >::New());\
  typedef Image<ipt,dm2> InputImageType2D;\
  typedef Image<opt,dm2> OutputImageType2D;\
  this->RegisterOverride(\
  typeid(BSplineDecompositionImageFilter<InputImageType2D,OutputImageType2D>).name(),\
  typeid(GPUBSplineDecompositionImageFilter<InputImageType2D,OutputImageType2D>).name(),\
  "GPU BSplineDecompositionImageFilter Override 2D",\
  true,\
  CreateObjectFunction<GPUBSplineDecompositionImageFilter<InputImageType2D,OutputImageType2D> >::New());\
  typedef Image<ipt,dm3> InputImageType3D;\
  typedef Image<opt,dm3> OutputImageType3D;\
  this->RegisterOverride(\
  typeid(BSplineDecompositionImageFilter<InputImageType3D,OutputImageType3D>).name(),\
  typeid(GPUBSplineDecompositionImageFilter<InputImageType3D,OutputImageType3D>).name(),\
  "GPU BSplineDecompositionImageFilter Override 3D",\
  true,\
  CreateObjectFunction<GPUBSplineDecompositionImageFilter<InputImageType3D,OutputImageType3D> >::New());\
  }

  GPUBSplineDecompositionImageFilterFactory()
  {
    if( IsGPUAvailable() )
    {
      // explicit type to float
      //OverrideBSplineDecompositionImageFilterTypeMacro(unsigned char, float, 1, 2, 3);
      //OverrideBSplineDecompositionImageFilterTypeMacro(char, float, 1, 2, 3);
      //OverrideBSplineDecompositionImageFilterTypeMacro(unsigned short, float, 1, 2, 3);
      OverrideBSplineDecompositionImageFilterTypeMacro(short, float, 1, 2, 3);
      //OverrideBSplineDecompositionImageFilterTypeMacro(unsigned int, float, 1, 2, 3);
      //OverrideBSplineDecompositionImageFilterTypeMacro(int, float, 1, 2, 3);
    }
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUBSplineDecompositionImageFilter.hxx"
#endif

#endif
