/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineInterpolateImageFunction_h
#define __itkGPUBSplineInterpolateImageFunction_h

#include "itkBSplineInterpolateImageFunction.h"
#include "itkGPUInterpolateImageFunction.h"
#include "itkVersion.h"

namespace itk
{
/** \class GPUBSplineInterpolateImageFunction
 */

/** Create a helper GPU Kernel class for GPUBSplineInterpolateImageFunction */
itkGPUKernelClassMacro(GPUBSplineInterpolateImageFunctionKernel);

template< class TInputImage, class TCoordRep = float, class TCoefficientType = float >
class ITK_EXPORT GPUBSplineInterpolateImageFunction
  : public GPUInterpolateImageFunction< TInputImage, TCoordRep,
  BSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType > >
{
public:
  /** Standard class typedefs. */
  typedef GPUBSplineInterpolateImageFunction  Self;
  typedef GPUInterpolateImageFunction< TInputImage, TCoordRep,
    BSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType > > GPUSuperclass;
  typedef BSplineInterpolateImageFunction< TInputImage, TCoordRep,
    BSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType > > CPUSuperclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  typedef typename GPUSuperclass::Superclass::CoefficientImageType CoefficientImageType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineInterpolateImageFunction, GPUSuperclass);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
    TInputImage::ImageDimension);

  typename CoefficientImageType::ConstPointer GetCoefficients() const {return this->m_Coefficients; };

protected:
  GPUBSplineInterpolateImageFunction();
  ~GPUBSplineInterpolateImageFunction() {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  virtual bool GetSourceCode(std::string &_source) const;

private:
  GPUBSplineInterpolateImageFunction(const Self &); //purposely not implemented
  void operator=(const Self &);                     //purposely not implemented

  std::vector<std::string> m_Sources;
  bool m_SourcesLoaded;
};

/** \class GPUBSplineInterpolateImageFunctionFactory
* \brief Object Factory implementation for GPUBSplineInterpolateImageFunction
*/
class GPUBSplineInterpolateImageFunctionFactory : public ObjectFactoryBase
{
public:
  typedef GPUBSplineInterpolateImageFunctionFactory Self;
  typedef ObjectFactoryBase        Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char* GetDescription() const { return "A Factory for GPUBSplineInterpolateImageFunction"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineInterpolateImageFunctionFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    GPUBSplineInterpolateImageFunctionFactory::Pointer factory = GPUBSplineInterpolateImageFunctionFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  GPUBSplineInterpolateImageFunctionFactory(const Self&); // purposely not implemented
  void operator=(const Self&);                                    // purposely not implemented

#define OverrideBSplineInterpolateImageFunctionTypeMacro(ipt,cr,ct,dm1,dm2,dm3)\
  {\
  typedef Image<ipt,dm1> InputImageType1D;\
  this->RegisterOverride(\
  typeid(BSplineInterpolateImageFunction<InputImageType1D,cr,ct>).name(),\
  typeid(GPUBSplineInterpolateImageFunction<InputImageType1D,cr,ct>).name(),\
  "GPU BSplineInterpolateImageFunction Override 1D",\
  true,\
  CreateObjectFunction<GPUBSplineInterpolateImageFunction<InputImageType1D,cr,ct> >::New());\
  typedef Image<ipt,dm2> InputImageType2D;\
  this->RegisterOverride(\
  typeid(BSplineInterpolateImageFunction<InputImageType2D,cr,ct>).name(),\
  typeid(GPUBSplineInterpolateImageFunction<InputImageType2D,cr,ct>).name(),\
  "GPU BSplineInterpolateImageFunction Override 2D",\
  true,\
  CreateObjectFunction<GPUBSplineInterpolateImageFunction<InputImageType2D,cr,ct> >::New());\
  typedef Image<ipt,dm3> InputImageType3D;\
  this->RegisterOverride(\
  typeid(BSplineInterpolateImageFunction<InputImageType3D,cr,ct>).name(),\
  typeid(GPUBSplineInterpolateImageFunction<InputImageType3D,cr,ct>).name(),\
  "GPU BSplineInterpolateImageFunction Override 3D",\
  true,\
  CreateObjectFunction<GPUBSplineInterpolateImageFunction<InputImageType3D,cr,ct> >::New());\
  }

  GPUBSplineInterpolateImageFunctionFactory()
  {
    if( IsGPUAvailable() )
    {
      // TCoordRep = float, TCoefficientType = float
      //OverrideBSplineInterpolateImageFunctionTypeMacro(unsigned char, float, float, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(char, float, float, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(unsigned short, float, float, 1, 2, 3);
      OverrideBSplineInterpolateImageFunctionTypeMacro(short, float, float, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(unsigned int, float, float, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(int, float, float, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(float, float, float, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(double, float, float, 1, 2, 3);

      // TCoordRep = double, TCoefficientType = double
      //OverrideBSplineInterpolateImageFunctionTypeMacro(unsigned char, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(char, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(unsigned short, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(short, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(unsigned int, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(int, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(float, double, double, 1, 2, 3);
      //OverrideBSplineInterpolateImageFunctionTypeMacro(double, double, double, 1, 2, 3);
    }
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUBSplineInterpolateImageFunction.hxx"
#endif

#endif
