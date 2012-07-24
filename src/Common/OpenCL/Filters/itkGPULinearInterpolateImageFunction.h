/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPULinearInterpolateImageFunction_h
#define __itkGPULinearInterpolateImageFunction_h

#include "itkLinearInterpolateImageFunction.h"
#include "itkGPUInterpolateImageFunction.h"

namespace itk
{
/** \class GPULinearInterpolateImageFunction
 */
template< class TInputImage, class TCoordRep = float >
class ITK_EXPORT GPULinearInterpolateImageFunction
  : public GPUInterpolateImageFunction< TInputImage, TCoordRep,
  LinearInterpolateImageFunction< TInputImage, TCoordRep > >
{
public:
  /** Standard class typedefs. */
  typedef GPULinearInterpolateImageFunction Self;
  typedef LinearInterpolateImageFunction< TInputImage, TCoordRep > CPUSuperclass;
  typedef GPUInterpolateImageFunction< TInputImage, TCoordRep, CPUSuperclass> GPUSuperclass;
  typedef SmartPointer< Self >              Pointer;
  typedef SmartPointer< const Self >        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPULinearInterpolateImageFunction, GPUSuperclass);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

protected:
  GPULinearInterpolateImageFunction();
  ~GPULinearInterpolateImageFunction() {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  virtual bool GetSourceCode(std::string &_source) const;

private:
  GPULinearInterpolateImageFunction(const Self &); //purposely not implemented
  void operator=(const Self &);                    //purposely not implemented

  std::vector<std::string> m_Sources;
  bool m_SourcesLoaded;
};

/** \class GPULinearInterpolateImageFunctionFactory
* \brief Object Factory implementation for GPULinearInterpolateImageFunction
*/
class GPULinearInterpolateImageFunctionFactory : public ObjectFactoryBase
{
public:
  typedef GPULinearInterpolateImageFunctionFactory Self;
  typedef ObjectFactoryBase        Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char* GetDescription() const { return "A Factory for GPULinearInterpolateImageFunction"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPULinearInterpolateImageFunctionFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    GPULinearInterpolateImageFunctionFactory::Pointer factory = GPULinearInterpolateImageFunctionFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  GPULinearInterpolateImageFunctionFactory(const Self&); // purposely not implemented
  void operator=(const Self&);                           // purposely not implemented

#define OverrideLinearInterpolateImageFunctionTypeMacro(ipt,opt,dm1,dm2,dm3)\
  {\
  typedef Image<ipt,dm1> InputImageType1D;\
  this->RegisterOverride(\
  typeid(LinearInterpolateImageFunction<InputImageType1D,opt>).name(),\
  typeid(GPULinearInterpolateImageFunction<InputImageType1D,opt>).name(),\
  "GPU LinearInterpolateImageFunction Override 1D",\
  true,\
  CreateObjectFunction<GPULinearInterpolateImageFunction<InputImageType1D,opt> >::New());\
  typedef Image<ipt,dm2> InputImageType2D;\
  this->RegisterOverride(\
  typeid(LinearInterpolateImageFunction<InputImageType2D,opt>).name(),\
  typeid(GPULinearInterpolateImageFunction<InputImageType2D,opt>).name(),\
  "GPU LinearInterpolateImageFunction Override 2D",\
  true,\
  CreateObjectFunction<GPULinearInterpolateImageFunction<InputImageType2D,opt> >::New());\
  typedef Image<ipt,dm3> InputImageType3D;\
  this->RegisterOverride(\
  typeid(LinearInterpolateImageFunction<InputImageType3D,opt>).name(),\
  typeid(GPULinearInterpolateImageFunction<InputImageType3D,opt>).name(),\
  "GPU LinearInterpolateImageFunction Override 3D",\
  true,\
  CreateObjectFunction<GPULinearInterpolateImageFunction<InputImageType3D,opt> >::New());\
  }

  GPULinearInterpolateImageFunctionFactory()
  {
    if( IsGPUAvailable() )
    {
      // TCoordRep = float
      //OverrideLinearInterpolateImageFunctionTypeMacro(unsigned char, float, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(char, float, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(unsigned short, float, 1, 2, 3);
      OverrideLinearInterpolateImageFunctionTypeMacro(short, float, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(unsigned int, float, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(int, float, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(float, float, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(double, float, 1, 2, 3);

      // TCoordRep = double
      //OverrideLinearInterpolateImageFunctionTypeMacro(unsigned char, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(char, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(unsigned short, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(short, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(unsigned int, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(int, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(float, double, 1, 2, 3);
      //OverrideLinearInterpolateImageFunctionTypeMacro(double, double, 1, 2, 3);
    }
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPULinearInterpolateImageFunction.hxx"
#endif

#endif
