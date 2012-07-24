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
#ifndef __itkGPUNearestNeighborInterpolateImageFunction_h
#define __itkGPUNearestNeighborInterpolateImageFunction_h

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkGPUInterpolateImageFunction.h"

namespace itk
{
/** \class GPUNearestNeighborInterpolateImageFunction
 */
template< class TInputImage, class TCoordRep = float >
class ITK_EXPORT GPUNearestNeighborInterpolateImageFunction
  : public GPUInterpolateImageFunction< TInputImage, TCoordRep,
  NearestNeighborInterpolateImageFunction< TInputImage, TCoordRep > >
{
public:
  /** Standard class typedefs. */
  typedef GPUNearestNeighborInterpolateImageFunction Self;
  typedef NearestNeighborInterpolateImageFunction< TInputImage, TCoordRep > CPUSuperclass;
  typedef GPUInterpolateImageFunction< TInputImage, TCoordRep, CPUSuperclass> GPUSuperclass;
  typedef SmartPointer< Self >                       Pointer;
  typedef SmartPointer< const Self >                 ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUNearestNeighborInterpolateImageFunction, GPUSuperclass);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

protected:
  GPUNearestNeighborInterpolateImageFunction();
  ~GPUNearestNeighborInterpolateImageFunction() {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  virtual bool GetSourceCode(std::string &_source) const;

private:
  GPUNearestNeighborInterpolateImageFunction(const Self &); //purposely not implemented
  void operator=(const Self &);                             //purposely not implemented

  std::vector<std::string> m_Sources;
  bool m_SourcesLoaded;
};

/** \class GPUNearestNeighborInterpolateImageFunctionFactory
* \brief Object Factory implementation for GPUNearestNeighborInterpolateImageFunction
*/
class GPUNearestNeighborInterpolateImageFunctionFactory : public ObjectFactoryBase
{
public:
  typedef GPUNearestNeighborInterpolateImageFunctionFactory Self;
  typedef ObjectFactoryBase        Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char* GetDescription() const { return "A Factory for GPUNearestNeighborInterpolateImageFunction"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUNearestNeighborInterpolateImageFunctionFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    GPUNearestNeighborInterpolateImageFunctionFactory::Pointer factory = GPUNearestNeighborInterpolateImageFunctionFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  GPUNearestNeighborInterpolateImageFunctionFactory(const Self&); // purposely not implemented
  void operator=(const Self&);                                    // purposely not implemented

#define OverrideNearestNeighborInterpolateImageFunctionTypeMacro(ipt,opt,dm1,dm2,dm3)\
  {\
  typedef Image<ipt,dm1> InputImageType1D;\
  this->RegisterOverride(\
  typeid(NearestNeighborInterpolateImageFunction<InputImageType1D,opt>).name(),\
  typeid(GPUNearestNeighborInterpolateImageFunction<InputImageType1D,opt>).name(),\
  "GPU NearestNeighborInterpolateImageFunction Override 1D",\
  true,\
  CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<InputImageType1D,opt> >::New());\
  typedef Image<ipt,dm2> InputImageType2D;\
  this->RegisterOverride(\
  typeid(NearestNeighborInterpolateImageFunction<InputImageType2D,opt>).name(),\
  typeid(GPUNearestNeighborInterpolateImageFunction<InputImageType2D,opt>).name(),\
  "GPU NearestNeighborInterpolateImageFunction Override 2D",\
  true,\
  CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<InputImageType2D,opt> >::New());\
  typedef Image<ipt,dm3> InputImageType3D;\
  this->RegisterOverride(\
  typeid(NearestNeighborInterpolateImageFunction<InputImageType3D,opt>).name(),\
  typeid(GPUNearestNeighborInterpolateImageFunction<InputImageType3D,opt>).name(),\
  "GPU NearestNeighborInterpolateImageFunction Override 3D",\
  true,\
  CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<InputImageType3D,opt> >::New());\
  }

  GPUNearestNeighborInterpolateImageFunctionFactory()
  {
    if( IsGPUAvailable() )
    {
      // TCoordRep = float
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(unsigned char, float, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(char, float, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(unsigned short, float, 1, 2, 3);
      OverrideNearestNeighborInterpolateImageFunctionTypeMacro(short, float, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(unsigned int, float, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(int, float, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(float, float, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(double, float, 1, 2, 3);

      // TCoordRep = double
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(unsigned char, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(char, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(unsigned short, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(short, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(unsigned int, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(int, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(float, double, 1, 2, 3);
      //OverrideNearestNeighborInterpolateImageFunctionTypeMacro(double, double, 1, 2, 3);
    }
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUNearestNeighborInterpolateImageFunction.hxx"
#endif

#endif
