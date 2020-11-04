// wek_ptr, shared_ptr and unique_ptr.
#include <memory>       // allocator, addressof
#include <atomic>       // atomic
#include <exception>    // exception
#include <type_traits>  // remove_extent, extent, remove_extent, is_array, is_void
                        // conditional, is_reference, common_type
#include <cstddef>      // nullptr_t, size_t, ptrdiff_t
#include <utility>      // move, forward, swap
#include <functional>   // less, hash
#include <iostream>     // basic_ostream, common_type
#include <tuple>        // tuple, get(tuple)
#include <cassert>      // assert

namespace sp {
    // Ptr class that wraps the deleter, use tuple for Empty Base Optimization
    template<typename T, typename D>
    class Ptr {
    public:
        constexpr Ptr() noexcept = default;
        Ptr(T* p) : _impl_t{} { _impl_ptr() = p; }
        template<typename Del>
        Ptr(T* p, Del&& d) : _impl_t{ p, std::forward<Del>(d) } { }
        ~Ptr() noexcept = default;

        T*& _impl_ptr() { return std::get<0>(_impl_t); }
        T* _impl_ptr() const { return std::get<0>(_impl_t); }
        D& _impl_deleter() { return std::get<1>(_impl_t); }
        const D& _impl_deleter() const { return std::get<1>(_impl_t); }

    private:
        std::tuple<T*, D> _impl_t;
    };

    // Control block interface, type erasure for storing deleter and allocators.
    class control_block_base {
    public:
        virtual ~control_block_base() { };

        virtual void inc_ref() noexcept = 0;
        virtual void inc_wref() noexcept = 0;
        virtual void dec_ref() noexcept = 0;
        virtual void dec_wref() noexcept = 0;

        virtual long use_count() const noexcept = 0;
        virtual bool unique() const noexcept = 0;
        virtual long weak_use_count() const noexcept = 0;
        virtual bool expired() const noexcept = 0;

        virtual void* get_deleter() noexcept = 0;
    };

    // Control block for reference counting of shared_ptr and weak_ptr.
    // No custom allocator support. The allocator is intended to be used 
    // to allocate/deallocate internal shared_ptr details, not the object.
    template<typename T, typename D = default_delete<T>>
    class control_block : public control_block_base {
    public:
        control_block(T* p) : _impl{ p } { }
        control_block(T* p, D d) : _impl{ p, d } { }
        ~control_block() { }

        void inc_ref() noexcept override { ++_use_count; }
        void inc_wref() noexcept override { ++_weak_use_count; }

        void dec_ref() noexcept override {
            auto _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (--_use_count == 0) {
                if (_ptr)
                    _deleter(_ptr); // destroy the object _ptr points to
                dec_wref();
            }
        }

        void dec_wref() noexcept override {
            if (--_weak_use_count == 0)
                delete this; // destroy control_block itself
        }

        // Return #shared_ptr
        long use_count() const noexcept override { return _use_count; }

        bool unique() const noexcept override { return _use_count == 1; }

        // Return #weak_ptr
        long weak_use_count() const noexcept override { return _weak_use_count - ((_use_count > 0) ? 1 : 0); }

        bool expired() const noexcept override { return _use_count == 0; }

        // Type erasure for storing deleter
        void* get_deleter() noexcept override { return reinterpret_cast<void*>(std::addressof(_impl._impl_deleter())); }

    private:
        std::atomic<long> _use_count{ 1 };
        // Note: _weak_use_count = #weak_ptrs + (#shared_ptr > 0) ? 1 : 0
        std::atomic<long> _weak_use_count{ 1 };
        Ptr<T, D> _impl;
    };

    // Default destruction policy used by unique_ptr & shared_ptr when no deleter is specified.
    template <typename T>
    class default_delete {
    public:
        // Default ctor.
        constexpr default_delete() noexcept = default;
        // Converting ctor, convertibility is not checked.
        template <typename U>
        default_delete(const default_delete<U>&) noexcept { }

        // Call operator
        void operator()(T* p) const { delete p; }
    };

    template <typename T>
    class default_delete<T[]> {
    public:
        // Default ctor.
        constexpr default_delete() noexcept = default;
        // Converting ctor, convertibility is not checked.
        template <typename U>
        default_delete(const default_delete<U[]>&) noexcept { }

        // Call operator.
        void operator()(T* p) const { delete[] p; }
    };

    // Type exception thrown by ctors of shared_ptr with weak_ptr as argument, when weak_ptr refers to already deleted object.
    class bad_weak_ptr : public std::exception {
    public:
        const char* what() noexcept { return "weak_ptr is expired!"; }
    };

    // Forward declaration
    template<typename T> class shared_ptr;

    // weak_ptr implementation
    template <typename T>
    class weak_ptr {
    public:
        template<typename U> friend class shared_ptr;
        template<typename U> friend class weak_ptr;

        using element_type = typename std::remove_extent<T>::type;

        // Default ctor, creates an empty weak_ptr.
        // Postconditions: use_count() == 0.
        constexpr weak_ptr() noexcept : _ptr{}, _control_block{} { }
        // Conversion ctor: shares ownership with sp.
        // Postconditions: use_count() == sp.use_count().
        template<class U>
        weak_ptr(shared_ptr<U> const& sp) noexcept : _ptr{ sp._ptr }, _control_block{ sp._control_block }
        { if (_control_block) _control_block->inc_wref(); }
        // Copy ctor: shares ownership with wp.
        // Postconditions: use_count() == wp.use_count().
        weak_ptr(weak_ptr const& wp) noexcept : _ptr{ wp._ptr }, _control_block{ wp._control_block }
        { if (_control_block) _control_block->inc_wref(); }
        // Copy ctor: shares ownership with wp.
        // Postconditions: use_count() == wp.use_count().
        template<class U>
        weak_ptr(weak_ptr<U> const& wp) noexcept : _ptr{ wp._ptr }, _control_block{ wp._control_block }
        { if (_control_block) _control_block->inc_wref(); }
        ~weak_ptr() { if (_control_block) _control_block->dec_wref(); }
        weak_ptr& operator=(const weak_ptr& wp) noexcept {
            weak_ptr{ wp }.swap(*this);
            return *this;
        }
        template<typename U>
        weak_ptr& operator=(const weak_ptr<U>& wp) noexcept {
            weak_ptr{ wp }.swap(*this);
            return *this;
        }
        template<typename U>
        weak_ptr& operator=(const shared_ptr<U>& sp) noexcept {
            weak_ptr{ sp }.swap(*this);
            return *this;
        }

        // Exchange contents of *this and sp.
        void swap(weak_ptr& wp) noexcept {
            using std::swap;
            swap(_ptr, wp._ptr);
            swap(_control_block, wp._control_block);
        }

        // Reset *this to empty.
        void reset() noexcept { weak_ptr{}.swap(*this); }

        // Get use_count
        long use_count() const noexcept { return (_control_block) ? _control_block->use_count() : 0; }

        // Check if use_count == 0
        bool expired() const noexcept { return (_control_block) ? _control_block->expired() : false; }

        // Check if there is a managed object
        shared_ptr<T> lock() const noexcept { return (expired()) ? shared_ptr<T>{} : shared_ptr<T>{ *this }; }

        // Check whether this shared_ptr precedes other in owner-based order
        // Implemented by comparing the address of control_block
        template<typename U>
        bool owner_before(shared_ptr<U> const& sp) const
        {
            return std::less<control_block_base*>()(_control_block, sp._control_block);
        }
        // Check whether this shared_ptr precedes other in owner-based order
        // Implemented by comparing the address of control_block
        template<class U>
        bool owner_before(weak_ptr<U> const& wp) const
        {
            return std::less<control_block_base*>()(_control_block, wp._control_block);
        }

    private:
        T* _ptr;
        control_block_base* _control_block;
    };

    // Swap with another weak_ptr
    template<typename T>
    inline void swap(weak_ptr<T>& wp1, weak_ptr<T>& wp2) { wp1.swap(wp2); }

    // Forward declarations.
    template<typename T, typename D> class unique_ptr;
    template<typename T> class shared_ptr;
    template<typename T> class weak_ptr;

    // Define operator*, operator-> and operator[] for T not array or cv void
    template<typename T, bool = std::is_array<T>::value, bool = std::is_void<T>::value>
    class shared_ptr_access {
    public:
        using element_type = T;

        // Dereference pointer to the managed object
        T& operator*() const noexcept { assert(_get() != nullptr); return *_get(); }
        T* operator->() const noexcept { assert(_get() != nullptr); return _get(); }

    private:
        T* _get() const noexcept { return static_cast<const shared_ptr<T>*>(this)->get(); }
    };

    // Specialization of shared_ptr_access for T array type. Defines operator[] for shared_ptr<T[]> and shared_ptr<T[N]>
    template<typename T>
    class shared_ptr_access<T, true, false> {
    public:
        using element_type = typename std::remove_extent<T>::type;

        // Index operator, dereferencing operators are not provided.
        T* operator[](std::ptrdiff_t i) const noexcept
        {
            assert(_get() != nullptr);
            static_assert(!std::extent<T>::value || i < std::extent<T>::value);
            return _get()[i];
        }

    private:
        T* _get() const noexcept { return static_cast<const shared_ptr<T>*>(this)->get(); }
    };

    // Specialization of shared_ptr_access for T cv void type. Defines operator-> for shared_ptr<cv void>
    template<typename T>
    class shared_ptr_access<T, false, true> {
    public:
        // Dereference pointer to the managed object, operator* is not provided
        T* operator->() const noexcept { assert(_get() != nullptr); return _get(); }

    private:
        T* _get() const noexcept { return static_cast<const shared_ptr<T>*>(this)->get(); }
    };

    // shared_ptr implementation.
    template<typename T>
    class shared_ptr : public shared_ptr_access<T> {
    public:
        template<typename U> friend class shared_ptr;
        template<typename U> friend class weak_ptr;
        template<typename D, typename U> friend D* get_deleter(const shared_ptr<U>&) noexcept;

        using element_type = typename shared_ptr_access<T>::element_type;
        using weak_type = weak_ptr<T>; /* added in C++17 */

        // Default ctor, creates a shared_ptr with no managed object
        // Postconditions: use_count() == 0 && get() == 0.
        constexpr shared_ptr() noexcept : _ptr{}, _control_block{} { }
        // Construct a shared_ptr with no managed object
        // Postconditions: use_count() == 0 && get() == 0.
        constexpr shared_ptr(std::nullptr_t) noexcept : _ptr{}, _control_block{} { }
        // Construct a shared_ptr with p as the pointer to the managed object
        // Postconditions: use_count() == 1 && get() == p. 
        template<typename U>
        explicit shared_ptr(U* p) : _ptr{ p }, _control_block{ new control_block<U>{p} } { }
        // Construct a shared_ptr with p as the pointer to the managed object, supplied with custom deleter
        // Postconditions: use_count() == 1 && get() == p.
        template<typename U, typename D>
        shared_ptr(U* p, D d) : _ptr{ p }, _control_block{ new control_block<U, D>{p, std::move(d)} } { }
        // Construct a shared_ptr with p as the pointer to the managed object, supplied with custom deleter and allocator
        // Postconditions: use_count() == 1 && get() == p.
        template<typename U, typename D, typename A>
        shared_ptr(U* p, D d, A a) = delete;
        // Construct a shared_ptr with no managed object, supplied with custom deleter
        // Postconditions: use_count() == 1 && get() == 0.
        template<typename D>
        shared_ptr(std::nullptr_t p, D d) : _ptr{ nullptr }, _control_block{ new control_block<T, D>{p, std::move(d)} } { }
        // Construct a shared_ptr with no managed object, supplied with custom deleter and allocator
        // Postconditions: use_count() == 1 && get() == 0.
        template<typename D, typename A>
        shared_ptr(std::nullptr_t p, D d, A a) = delete;
        // Aliasing ctor: constructs a shared_ptr instance that stores p and shares ownership with sp
        // Postconditions: use_count() == sp.use_count() && get() == p.
        template<typename U>
        shared_ptr(const shared_ptr<U>& sp, T* p) noexcept : _ptr{ p }, _control_block{ sp._control_block }
        { if (_control_block) _control_block->inc_ref(); }
        // Copy ctor: shares ownership of the object managed by sp
        // Postconditions: use_count() == sp.use_count() && get() == sp.get().
        shared_ptr(const shared_ptr& sp) noexcept : _ptr{ sp._ptr }, _control_block{ sp._control_block }
        { if (_control_block) _control_block->inc_ref(); }
        // Copy ctor: shares ownership of the object managed by sp
        // Postconditions: use_count() == sp.use_count() && get() == sp.get().
        template<typename U>
        shared_ptr(const shared_ptr<U>& sp) noexcept : _ptr{ sp._ptr }, _control_block{ sp._control_block }
        { if (_control_block) _control_block->inc_ref(); }
        // Move ctor: Move-constructs a shared_ptr from sp
        // Postconditions: *this shall contain the old value of sp.
        // sp shall be empty. sp.get() == 0.
        shared_ptr(shared_ptr&& sp) noexcept : _ptr{ std::move(sp._ptr) }, _control_block{ std::move(sp._control_block) } {
            sp._ptr = nullptr;
            sp._control_block = nullptr;
        }
        // Move ctor: Move-constructs a shared_ptr from sp
        // Postconditions: *this shall contain the old value of sp.
        //     sp shall be empty. sp.get() == 0.
        template<typename U>
        shared_ptr(shared_ptr<U>&& sp) noexcept : _ptr{ sp._ptr }, _control_block{ sp._control_block } {
            sp._ptr = nullptr;
            sp._control_block = nullptr;
        }
        // Constructsa shared_ptr object that shares ownership with wp
        // Postconditions: use_count() == wp.use_count().
        template<typename U>
        explicit shared_ptr(const weak_ptr<U>& wp) : _ptr{ wp._ptr }, _control_block{ wp._control_block } {
            if (wp.expired())
                throw bad_weak_ptr{};
            else
                _control_block->inc_ref();
        }
        // Construct a shared_ptr object that obtains ownership from up
        // Postconditions: use_count() == 1. up shall be empty. up.get() = 0.
        template<typename U, typename D>
        shared_ptr(unique_ptr<U, D>&& up) : shared_ptr{ up.release(), up.get_deleter() } { }
        ~shared_ptr() { if (_control_block) _control_block->dec_ref(); }
        // Copy assignment
        shared_ptr& operator=(const shared_ptr& sp) noexcept {
            shared_ptr{ sp }.swap(*this);
            return *this;
        }
        template<typename U>
        shared_ptr& operator=(const shared_ptr<U>& sp) noexcept {
            shared_ptr{ sp }.swap(*this);
            return *this;
        }
        // Move assignment
        shared_ptr& operator=(shared_ptr&& sp) noexcept {
            shared_ptr{ std::move(sp) }.swap(*this);
            return *this;
        }
        template<typename U>
        shared_ptr& operator=(shared_ptr<U>&& sp) noexcept {
            shared_ptr{ std::move(sp) }.swap(*this);
            return *this;
        }
        // Move assignment from a unique_ptr
        template<typename U, typename D>
        shared_ptr& operator=(unique_ptr<U, D>&& up) noexcept {
            shared_ptr{ std::move(up) }.swap(*this);
            return *this;
        }

        // Exchange the contents of *this and sp
        void swap(shared_ptr& sp) noexcept {
            using std::swap;
            swap(_ptr, sp._ptr);
            swap(_control_block, sp._control_block);
        }

        // Reset *this to empty
        void reset() noexcept { shared_ptr{}.swap(*this); }
        // Reset *this with p as the pointer to the managed object
        template<typename U>
        void reset(U* p) { shared_ptr{ p }.swap(*this); }
        // Reset *this with p as the pointer to the managed object, supplied with custom deleter
        template<typename U, typename D>
        void reset(U* p, D d) { shared_ptr{ p, d }.swap(*this); }
        // Reset *this with p as the pointer to the managed object, supplied with custom deleter and allocator
        template<typename U, typename D, typename A>
        void reset(U* p, D d, A a) = delete;

        // Get the stored pointer
        T* get() const noexcept { return _ptr; }

        // Get use_count
        long use_count() const noexcept { return (_control_block) ? _control_block->use_count() : 0; }

        // deprecated in C++17, removed in C++20
        // Check if use_count == 1
        bool unique() const noexcept { return (_control_block) ? _control_block->unique() : false; }

        // Check if there is a managed object
        explicit operator bool() const noexcept { return (_ptr) ? true : false; }

        // Check whether this shared_ptr precedes other in owner-based order
        // Implemented by comparing the address of control_block
        template<typename U>
        bool owner_before(shared_ptr<U> const& sp) const {
            return std::less<control_block_base*>()(_control_block, sp._control_block);
        }

        // Check whether this shared_ptr precedes other in owner-based order
        // Implemented by comparing the address of control_block
        template<class U>
        bool owner_before(weak_ptr<U> const& wp) const {
            return std::less<control_block_base*>()(_control_block, wp._control_block);
        }

    private:
        T* _ptr;
        control_block_base* _control_block;
    };

    // Create a shared_ptr that manages a new object.
    template<typename T, typename... Args>
    inline shared_ptr<T> make_shared(Args&&... args) { return shared_ptr<T>{new T{ std::forward<Args>(args)... }}; }

    template<typename T, typename A, typename... Args>
    inline shared_ptr<T> allocate_shared(const A& a, Args&&... args) = delete;

    // Operator overloading.
    template<typename T, typename U>
    inline bool operator==(const shared_ptr<T>& sp1, const shared_ptr<U>& sp2) { return sp1.get() == sp2.get(); }
    template<typename T>
    inline bool operator==(const shared_ptr<T>& sp, std::nullptr_t) noexcept { return !sp; }
    template<typename T>
    inline bool operator==(std::nullptr_t, const shared_ptr<T>& sp) noexcept { return !sp; }

    template<typename T, typename U>
    inline bool operator!=(const shared_ptr<T>& sp1, const shared_ptr<U>& sp2) { return sp1.get() != sp2.get(); }
    template<typename T>
    inline bool operator!=(const shared_ptr<T>& sp, std::nullptr_t) noexcept { return bool{ sp }; }
    template<typename T>
    inline bool operator!=(std::nullptr_t, const shared_ptr<T>& sp) noexcept { return bool{ sp }; }

    template<typename T, typename U>
    inline bool operator<(const shared_ptr<T>& sp1, const shared_ptr<U>& sp2) {
        using _Tp_elt = typename shared_ptr<T>::element_type;
        using _Up_elt = typename shared_ptr<U>::element_type;
        using _CT = typename std::common_type<_Tp_elt*, _Up_elt*>::type;
        return std::less<_CT>()(sp1.get(), sp2.get());
    }
    template<typename T>
    inline bool operator<(const shared_ptr<T>& sp, std::nullptr_t) {
        using _Tp_elt = typename shared_ptr<T>::element_type;
        return std::less<_Tp_elt*>()(sp.get(), nullptr);
    }
    template<typename T>
    inline bool operator<(std::nullptr_t, const shared_ptr<T>& sp) {
        using _Tp_elt = typename shared_ptr<T>::element_type;
        return std::less<_Tp_elt*>()(nullptr, sp.get());
    }

    template<typename T, typename U>
    inline bool operator<=(const shared_ptr<T>& sp1, const shared_ptr<U>& sp2) { return !(sp2.get() < sp1.get()); }
    template<typename T>
    inline bool operator<=(const shared_ptr<T>& sp, std::nullptr_t) { return !(nullptr < sp.get()); }
    template<typename T>
    inline bool operator<=(std::nullptr_t, const shared_ptr<T>& sp) { return !(sp.get() < nullptr); }

    template<typename T, typename U>
    inline bool operator>(const shared_ptr<T>& sp1, const shared_ptr<U>& sp2) { return sp2.get() < sp1.get(); }
    template<typename T>
    inline bool operator>(const shared_ptr<T>& sp, std::nullptr_t) { return nullptr < sp.get(); }
    template<typename T>
    inline bool operator>(std::nullptr_t, const shared_ptr<T>& sp) { return sp.get() < nullptr; }

    template<typename T, typename U>
    inline bool operator>=(const shared_ptr<T>& sp1, const shared_ptr<U>& sp2) { return !(sp1.get() < sp2.get()); }
    template<typename T>
    inline bool operator>=(const shared_ptr<T>& sp, std::nullptr_t) { return !(sp.get() < nullptr); }
    template<typename T>
    inline bool operator>=(std::nullptr_t, const shared_ptr<T>& sp) { return !(nullptr < sp.get()); }

    // Swap with another shared_ptr.
    template<typename T>
    inline void swap(shared_ptr<T>& sp1, shared_ptr<T>& sp2) { sp1.swap(sp2); }

    // shared_ptr casts
    template<typename T, typename U>
    inline shared_ptr<T> static_pointer_cast(const shared_ptr<U>& sp) noexcept {
        using _Sp = shared_ptr<T>;
        return _Sp(sp, static_cast<typename _Sp::element_type*>(sp.get()));
    }
    template<typename T, typename U>
    inline shared_ptr<T> const_pointer_cast(const shared_ptr<U>& sp) noexcept {
        using _Sp = shared_ptr<T>;
        return _Sp(sp, const_cast<typename _Sp::element_type*>(sp.get()));
    }
    template<typename T, typename U>
    inline shared_ptr<T> dynamic_pointer_cast(const shared_ptr<U>& sp) noexcept {
        using _Sp = shared_ptr<T>;
        if (auto* _p = dynamic_cast<typename _Sp::element_type*>(sp.get()))
            return _Sp(sp, _p);
        return _Sp();
    }
    // Added in C++17 
    template<typename T, typename U>
    inline shared_ptr<T> reinterpret_pointer_cast(const shared_ptr<U>& sp) noexcept {
        using _Sp = shared_ptr<T>;
        return _Sp(sp, reinterpret_cast<typename _Sp::element_type*>(sp.get()));
    }

    // shared_ptr get_deleter
    template<typename D, typename T>
    inline D* get_deleter(const shared_ptr<T>& sp) noexcept { return reinterpret_cast<D*>(sp._control_block->get_deleter()); }

    // shared_ptr I/O
    template<class E, class T, class Y>
    inline std::basic_ostream<E, T>&
        operator<<(std::basic_ostream<E, T>& os, const shared_ptr<Y>& sp) {
        os << sp.get();
        return os;
    }

    // unique_ptr for single objects.
    template<typename T, typename D = default_delete<T>>
    class unique_ptr {
    public:
        using pointer = T*;
        using deleter_type = D;

        // Default ctor, creates a unique_ptr that owns nothing.
        constexpr unique_ptr() noexcept = default;
        // Construct with nullptr, creates a unique_ptr that owns nothing
        constexpr unique_ptr(std::nullptr_t) noexcept { }
        // Take ownership from a pointer
        explicit unique_ptr(T* p) noexcept : _impl{ p } { }
        // Take ownership from a pointer, supplied with a custom deleter
        // d: a reference to a deleter.
        unique_ptr(T* p, typename std::conditional<std::is_reference<deleter_type>::value, deleter_type, const deleter_type&>::type d) noexcept : _impl{ p, d } { }
        // Take ownership from a pointer, supplied with a custom deleter
        // d: an rvalue reference to a deleter.
        // Not permitted if deleter_type is an lvalue reference.
        unique_ptr(T* p, typename std::remove_reference<deleter_type>::type&& d) noexcept : _impl{ p, std::move(d) } 
        { static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference"); }
        // Move ctor: takes ownership from a unique_ptr of the same type
        unique_ptr(unique_ptr&& up) noexcept : _impl{ up.release(), std::forward<deleter_type>(up.get_deleter()) } { }
        // Move ctor: takes ownership from a unique_ptr of a different type
        template <typename U, typename E>
        unique_ptr(unique_ptr<U, E>&& up) noexcept : _impl{ up.release(), std::forward<deleter_type>(up.get_deleter()) } { }
        // Invoke the deleter if the stored pointer is not null
        ~unique_ptr() noexcept {
            auto _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (_ptr)
                _deleter(_ptr);
        }
        // Move assignment: takes ownership from a unique_ptr of the same type
        unique_ptr& operator=(unique_ptr&& up) noexcept {
            reset(up.release());
            auto& _deleter = _impl._impl_deleter();
            _deleter = up.get_deleter();
            return *this;
        }
        // Move assignment: takes ownership from a unique_ptr of a different type
        template <typename U, typename E>
        unique_ptr& operator=(unique_ptr<U, E>&& up) noexcept {
            reset(up.release());
            auto& _deleter = _impl._impl_deleter();
            _deleter = up.get_deleter();
            return *this;
        }
        // Reset unique_ptr to empty if assigned to nullptr 
        unique_ptr& operator=(std::nullptr_t) noexcept {
            reset();
            return *this;
        }

        // Dereference pointer to the managed object
        T& operator*() const noexcept {
            auto _ptr = _impl._impl_ptr();
            assert(_ptr != nullptr);
            return *_ptr;
        }
        T* operator->() const noexcept {
            auto _ptr = _impl._impl_ptr();
            assert(_ptr != nullptr);
            return _ptr;
        }

        // Get the stored pointer
        T* get() const noexcept { return _impl._impl_ptr(); }

        // Get a reference to the stored deleter
        deleter_type& get_deleter() noexcept { return _impl._impl_deleter(); }

        // Get a const reference to the stored deleter
        const deleter_type& get_deleter() const noexcept { return _impl._impl_deleter(); }

        // Check if there is an associated managed object
        explicit operator bool() const noexcept { return (_impl._impl_ptr()) ? true : false; }

        // Release ownership to the returned raw pointer
        T* release() noexcept {
            auto& _ptr = _impl._impl_ptr();
            T* cp = _ptr;
            _ptr = nullptr;
            return cp;
        }

        // Reset unique_ptr to empty and takes ownership from a pointer
        void reset(T* p) noexcept {
            auto& _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (_ptr)
                _deleter(_ptr);
            _ptr = p;
        }
        // Reset unique_ptr to empty
        void reset() noexcept {
            auto& _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (_ptr)
                _deleter(_ptr);
            _ptr = pointer{}; // probably nullptr
        }

        // Swap with another unique_ptr
        void swap(unique_ptr& up) noexcept {
            using std::swap;
            swap(_impl, up._impl);
        }

        // Disable copy from lvalue, disable copy constructor
        unique_ptr(const unique_ptr&) = delete;
        // Disable copy assignment
        unique_ptr& operator=(const unique_ptr&) = delete;

    private:
        Ptr<T, D> _impl;
    };

    // unique_ptr for array objects with a runtime length
    template <typename T, typename D>
    class unique_ptr<T[], D> {
    public:
        using pointer = T*;
        using deleter_type = D;

        // Default ctor, creates a unique_ptr that owns nothing
        constexpr unique_ptr() noexcept = default;
        // Construct with nullptr, creates a unique_ptr that owns nothing
        constexpr unique_ptr(std::nullptr_t) noexcept { }
        // Take ownership from a pointer
        explicit unique_ptr(T* p) noexcept : _impl{ p } { }
        // Take ownership from a pointer, supplied with a custom deleter
        // d: a reference to a deleter.
        unique_ptr(T* p, typename std::conditional<std::is_reference<deleter_type>::value, deleter_type, const deleter_type&>::type d) noexcept : _impl{ p, d } { }
        // Take ownership from a pointer, supplied with a custom deleter
        // d: an rvalue reference to a deleter.
        // Not permitted if deleter_type is an lvalue reference.
        unique_ptr(T* p, typename std::remove_reference<deleter_type>::type&& d) noexcept : _impl{ p, std::move(d) } 
        { static_assert(!std::is_reference<deleter_type>::value, "rvalue deleter bound to reference"); }
        // Move ctor: takes ownership from a unique_ptr of the same type
        unique_ptr(unique_ptr&& up) noexcept : _impl{ up.release(), up.get_deleter() } { }
        // Invoke deleter if the stored pointer is not null
        ~unique_ptr() noexcept {
            auto _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (_ptr)
                _deleter(_ptr);
        }
        // Move assignment: takes ownership from a unique_ptr of the same type
        unique_ptr& operator= (unique_ptr&& up) noexcept {
            reset(up.release());
            auto& _deleter = _impl._impl_deleter();
            _deleter = up.get_deleter();
            return *this;
        }
        // Reset unique_ptr to empty if assigned to nullptr 
        unique_ptr& operator=(std::nullptr_t) noexcept {
            reset();
            return *this;
        }

        // Index operator, dereferencing operators are not provided, bound range is not checked
        T& operator[](std::size_t i) const noexcept {
            auto _ptr = _impl._impl_ptr();
            assert(_ptr != nullptr);
            return _ptr[i];
        }

        // Get the stored pointer
        T* get() const noexcept {
            return _impl._impl_ptr();
        }  
        // Get a reference to the stored delete
        deleter_type& get_deleter() noexcept { return _impl._impl_deleter(); }
        // Get a const reference to the stored deleter
        const deleter_type& get_deleter() const noexcept { return _impl._impl_deleter(); }

        // Check if there is an associated managed object
        explicit operator bool() const noexcept { return (_impl._impl_ptr()) ? true : false; }

        // Release ownership to the returned raw pointer
        T* release() noexcept {
            auto& _ptr = _impl._impl_ptr();
            T* cp = _ptr;
            _ptr = nullptr;
            return cp;
        }

        // Reset unique_ptr to empty and takes ownership from a pointer
        void reset(T* p) noexcept {
            auto& _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (_ptr)
                _deleter(_ptr);
            _ptr = p;
        }
        // Reset unique_ptr to empty
        void reset() noexcept {
            auto& _ptr = _impl._impl_ptr();
            auto& _deleter = _impl._impl_deleter();
            if (_ptr)
                _deleter(_ptr);
            _ptr = pointer{}; // probably nullptr
        }

        // Swap with another unique_ptr
        void swap(unique_ptr& up) noexcept {
            using std::swap;
            swap(_impl, up._impl);
        }

        // Disable copy from lvalue, disable copy constructor
        unique_ptr(const unique_ptr&) = delete;
        // Disable copy assignment
        unique_ptr& operator=(const unique_ptr&) = delete;

    private:
        Ptr<T, D> _impl;
    };

    // make_unique: create a unique pointer that manages a new object. added in C++14.
    template<typename T>
    struct _Unique_if { using _Single_object = unique_ptr<T>; };
    template<typename T>
    struct _Unique_if<T[]> { using _Unknown_bound = unique_ptr<T[]>; };
    template<typename T, std::size_t N>
    struct _Unique_if<T[N]> { using _Known_bound = void; };

    // Only for non-array types
    template<typename T, typename... Args>
    typename _Unique_if<T>::_Single_object make_unique(Args&&... args)
    {
        return unique_ptr<T>{new T{ std::forward<Args>(args)... }};
    }
    // Only for array types with unknown bound
    template<typename T>
    typename _Unique_if<T>::_Unknown_bound make_unique(std::size_t n) {
        using U = typename std::remove_extent<T>::type;
        return unique_ptr<T>{new U[n]{}};
    }
    // Only for array types with known bound: unspecified
    template<typename T, typename... Args>
    typename _Unique_if<T>::_Known_bound make_unique(Args&&...) = delete;

    // Operator overloading.
    template<typename T, typename D, typename U, typename E>
    inline bool operator==(const unique_ptr<T, D>& up1, const unique_ptr<U, E>& up2) { return up1.get() == up2.get(); }
    template<typename T, typename D>
    inline bool operator==(const unique_ptr<T, D>& up, std::nullptr_t) noexcept { return !up; }
    template<typename T, typename D>
    inline bool operator==(std::nullptr_t, const unique_ptr<T, D>& up) noexcept { return !up; }

    template<typename T, typename D, typename U, typename E>
    inline bool operator!=(const unique_ptr<T, D>& up1, const unique_ptr<U, E>& up2) { return up1.get() != up2.get(); }
    template<typename T, typename D>
    inline bool operator!=(const unique_ptr<T, D>& up, std::nullptr_t) noexcept { return bool{ up }; }
    template<typename T, typename D>
    inline bool operator!=(std::nullptr_t, const unique_ptr<T, D>& up) noexcept { return bool{ up }; }

    template<typename T, typename D, typename U, typename E>
    inline bool operator<(const unique_ptr<T, D>& up1, const unique_ptr<U, E>& up2) {
        using _CT = typename std::common_type<typename unique_ptr<T, D>::pointer, typename unique_ptr<U, E>::pointer>::type;
        return std::less<_CT>()(up1.get(), up2.get());
    }
    template<typename T, typename D>
    inline bool operator<(const unique_ptr<T, D>& up, std::nullptr_t) {
        return std::less<typename unique_ptr<T, D>::pointer>()(up.get(), nullptr);
    }
    template<typename T, typename D>
    inline bool operator<(std::nullptr_t, const unique_ptr<T, D>& up) {
        return std::less<typename unique_ptr<T, D>::pointer>()(nullptr, up.get());
    }

    template<typename T, typename D, typename U, typename E>
    inline bool operator<=(const unique_ptr<T, D>& up1, const unique_ptr<U, E>& up2) { return !(up2.get() < up1.get()); }
    template<typename T, typename D>
    inline bool operator<=(const unique_ptr<T, D>& up, std::nullptr_t) { return !(nullptr < up.get()); }
    template<typename T, typename D>
    inline bool operator<=(std::nullptr_t, const unique_ptr<T, D>& up) { return !(up.get() < nullptr); }

    template<typename T, typename D, typename U, typename E>
    inline bool operator>(const unique_ptr<T, D>& up1, const unique_ptr<U, E>& up2) { return up2.get() < up1.get(); }
    template<typename T, typename D>
    inline bool operator>(const unique_ptr<T, D>& up, std::nullptr_t) { return nullptr < up.get(); }
    template<typename T, typename D>
    inline bool operator>(std::nullptr_t, const unique_ptr<T, D>& up) { return up.get() < nullptr; }

    template<typename T, typename D, typename U, typename E>
    inline bool operator>=(const unique_ptr<T, D>& up1, const unique_ptr<U, E>& up2) { return !(up1.get() < up2.get()); }
    template<typename T, typename D>
    inline bool operator>=(const unique_ptr<T, D>& up, std::nullptr_t) { return !(up.get() < nullptr); }
    template<typename T, typename D>
    inline bool operator>=(std::nullptr_t, const unique_ptr<T, D>& up) { return !(nullptr < up.get()); }

    // Swap with another unique_ptr
    template<typename T, typename D>
    inline void swap(unique_ptr<T, D>& up1, unique_ptr<T, D>& up2) { up1.swap(up2); }

    // unique_ptr I/O added in C++20 
    template<class E, class T, class Y>
    std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, const unique_ptr<Y>& up) {
        os << up.get();
        return os;
    }

    // enable_shared_from_this allows an object t that is currently managed by a shared_ptr named sp to safely generate additional shared_ptr instances
    //  sp1, sp2, ... that all share ownership of t with sp.
    // Publicly inheriting from enable_shared_from_this<T> provides the type T with a member function shared_from_this. If an object t of type T is
    //  managed by a shared_ptr<T> named sp, then calling T::shared_from_this will return a new shared_ptr<T> that shares ownership of t with sp.
    // Forward declarations.
    template<typename T> class shared_ptr;
    template<typename T> class weak_ptr;

    template<typename T>
    class enable_shared_from_this {
    private:
        weak_ptr<T> weak_this;

    protected:
        constexpr enable_shared_from_this() noexcept : weak_this{} { }
        enable_shared_from_this(const enable_shared_from_this& r) noexcept { }
        enable_shared_from_this& operator=(const enable_shared_from_this&) { return *this; }
        ~enable_shared_from_this() { }

    public:
        shared_ptr<T> shared_from_this() { return shared_ptr<T>(weak_this); }
        shared_ptr<const T> shared_from_this() const { return shared_ptr<const T>(weak_this); }
    };
} // namespace smart_ptr
