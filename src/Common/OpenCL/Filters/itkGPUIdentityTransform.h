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
#ifndef __itkGPUIdentityTransform_h
#define __itkGPUIdentityTransform_h

#include "itkIdentityTransform.h"
#include "itkGPUTransformBase.h"

namespace itk
{
/** \class GPUIdentityTransform
 */
template<class TScalarType = float, unsigned int NDimensions = 3,
class TParentImageFilter = IdentityTransform< TScalarType, NDimensions > >
class GPUIdentityTransform : public TParentImageFilter, public GPUTransformBase
{
public:
  /** Standard class typedefs. */
  typedef GPUIdentityTransform        Self;
  typedef TParentImageFilter          Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUIdentityTransform, TParentImageFilter);

protected:
  GPUIdentityTransform();
  virtual ~GPUIdentityTransform() {};
  void PrintSelf(std::ostream &s, Indent indent) const;

  virtual bool GetSourceCode(std::string &_source) const;

private:
  GPUIdentityTransform(const Self & other); // purposely not implemented
  const Self & operator=(const Self &);     // purposely not implemented

  std::vector<std::string> m_Sources;
  bool m_SourcesLoaded;
};

/** \class GPUIdentityTransformFactory
* \brief Object Factory implementation for GPUIdentityTransform
*/
class GPUIdentityTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUIdentityTransformFactory Self;
  typedef ObjectFactoryBase         Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char* GetDescription() const { return "A Factory for GPUIdentityTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUIdentityTransformFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    GPUIdentityTransformFactory::Pointer factory = GPUIdentityTransformFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  GPUIdentityTransformFactory(const Self&); // purposely not implemented
  void operator=(const Self&);            // purposely not implemented

#define OverrideIdentityTransformTypeMacro(st,dm)\
  {\
  this->RegisterOverride(\
  typeid(IdentityTransform<st,dm>).name(),\
  typeid(GPUIdentityTransform<st,dm>).name(),\
  "GPU IdentityTransform Override",\
  true,\
  CreateObjectFunction<GPUIdentityTransform<st,dm> >::New());\
  }

  GPUIdentityTransformFactory()
  {
    if( IsGPUAvailable() )
    {
      OverrideIdentityTransformTypeMacro(float, 1);
      //OverrideIdentityTransformTypeMacro(double, 1);

      OverrideIdentityTransformTypeMacro(float, 2);
      //OverrideIdentityTransformTypeMacro(double, 2);

      OverrideIdentityTransformTypeMacro(float, 3);
      //OverrideIdentityTransformTypeMacro(double, 3);
    }
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUIdentityTransform.hxx"
#endif

#endif /* __itkGPUIdentityTransform_h */
